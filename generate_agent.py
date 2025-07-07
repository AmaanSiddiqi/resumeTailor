#!/usr/bin/env python3
"""
generate_agent.py - Enhanced Resume Tailoring Agent

Intelligently tailors your LaTeX resume to job descriptions with:
 1. Advanced keyword extraction (TF-IDF + named entity recognition)
 2. Smart resume block selection with relevance scoring
 3. Context-aware bullet point rewriting via LLM
 4. Comprehensive LaTeX rendering with error handling
 5. Automated PDF compilation with cleanup
 6. Caching and performance optimizations
 7. Comprehensive error handling and validation

Usage:
    python generate_agent.py --jd job_description.txt [options]
    python generate_agent.py --jd job_description.txt --interactive
    python generate_agent.py --help
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm # For the progress bar in the next step
from enum import Enum
import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# LLM dependencies
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Some features may be limited.")

# Optional NLP enhancements
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ————— Data Classes —————
class RewriteStyle(str, Enum):
    TECHNICAL = "technical"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"

class RewriteFocus(str, Enum):
    ATS = "ats"
    HUMAN = "human"
    BALANCED = "balanced"

class ResumeSection(str, Enum):
    EXPERIENCE = "Experience"
    PROJECTS = "Projects"
    EDUCATION = "Education"
    SKILLS = "Skills" # While we handle skills separately, it's good to have it defined.

@dataclass
class ResumeBlock:
    """A structured block of resume content (e.g., a job or project)."""
    heading: str
    # MODIFIED: Use the ResumeSection Enum for type safety and clarity.
    section: ResumeSection
    bullets: List[str]
    date: str = ""
    tags: List[str] = field(default_factory=list)
    priority: int = 0  # User-defined importance (0-10) for manual ranking.
    relevance_score: float = 0.0  # Calculated during scoring.

    # NEW: A helper property to get all text content from the block easily.
    @property
    def full_text(self) -> str:
        """Returns all text content of the block for analysis."""
        return f"{self.heading} {' '.join(self.bullets)} {' '.join(self.tags)}".lower()


@dataclass
class JobAnalysis:
    """Parsed data extracted from a job description."""
    job_title: str
    keywords: List[str]
    required_skills: List[str]
    preferred_skills: List[str]
    company: Optional[str] = None # Use None as the default for optional fields
    seniority_level: Optional[str] = None
    # NEW: Store the original text for potential future use (e.g., passing full context to an LLM).
    raw_text: str = ""


@dataclass
class GenerationConfig:
    """
    Centralized configuration for the entire generation process.
    This is the single source of truth for all operational parameters.
    """
    # LLM and Rewriting Config
    model: str
    temperature: float
    rewrite_style: RewriteStyle
    focus: RewriteFocus

    # Resume Structure Config
    max_experience_blocks: int
    max_project_blocks: int
    max_bullets_per_block: int

    # Caching Config
    use_cache: bool
    cache_ttl: int

    # NEW: A class method to create the config directly from the parsed arguments.
    # This centralizes the logic of mapping CLI args to the config object.
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> GenerationConfig:
        return cls(
            model=args.model,
            temperature=args.temperature,
            rewrite_style=RewriteStyle(args.style), # Convert string to Enum
            focus=RewriteFocus(args.focus),         # Convert string to Enum
            max_experience_blocks=args.max_experience_blocks, # NEW: More granular control
            max_project_blocks=args.max_project_blocks,     # NEW: More granular control
            max_bullets_per_block=args.max_bullets,
            use_cache=not args.no_cache,
            cache_ttl=3600 # Can be made a CLI arg later if needed
        )

# ————— Enhanced Configuration & CLI —————

# NEW: Define default configuration values in a single dictionary.
# This makes it easy to see all defaults at a glance.
DEFAULT_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "max_experience_blocks": 3,
    "max_project_blocks": 3,
    "max_bullets_per_block": 4,
    "style": RewriteStyle.TECHNICAL,
    "focus": RewriteFocus.BALANCED,
}

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the resume tailoring system."""
    parser = argparse.ArgumentParser(
        description="✨ AI-powered resume tailoring system ✨",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python %(prog)s --jd job.txt --verbose
  python %(prog)s --jd job.txt --interactive --max-experience 2 --max-projects 4
  python %(prog)s --jd job.txt --style professional --model gpt-4
  python %(prog)s --jd job.txt --dry-run --cache-clear
        """
    )

    # Use argument groups for a cleaner --help message
    io_group = parser.add_argument_group("Input/Output")
    gen_group = parser.add_argument_group("Resume Generation Options")
    llm_group = parser.add_argument_group("LLM Configuration")
    exec_group = parser.add_argument_group("Execution & Performance")

    # --- Input/Output Arguments ---
    io_group.add_argument("-j", "--jd", type=Path, required=True, help="Path to job description text file.")
    io_group.add_argument("-b", "--blocks", type=Path, default=Path("blocks_dynamic.yaml"), help="YAML file with resume content.")
    io_group.add_argument("-t", "--template-dir", type=Path, default=Path("templates"), help="Directory with LaTeX Jinja2 templates.")
    io_group.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Directory to save outputs (PDF/Tex).")
    io_group.add_argument("--output-name", type=str, default="tailored_resume", help="Base filename for output (without extension).")

    # --- Resume Generation Options ---
    gen_group.add_argument("--max-experience-blocks", type=int, default=DEFAULT_CONFIG["max_experience_blocks"], help="Maximum 'Experience' blocks to include.")
    gen_group.add_argument("--max-project-blocks", type=int, default=DEFAULT_config["max_project_blocks"], help="Maximum 'Projects' blocks to include.")
    gen_group.add_argument("--max-bullets", dest="max_bullets_per_block", type=int, default=DEFAULT_CONFIG["max_bullets_per_block"], help="Maximum bullets per block.")
    gen_group.add_argument("--style", type=RewriteStyle, choices=list(RewriteStyle), default=DEFAULT_CONFIG["style"], help="LLM rewrite style for bullets.")
    gen_group.add_argument("--focus", type=RewriteFocus, choices=list(RewriteFocus), default=DEFAULT_CONFIG["focus"], help="LLM rewrite focus (ATS vs. human).")

    # --- LLM Configuration ---
    llm_group.add_argument("-m", "--model", type=str, default=DEFAULT_CONFIG["model"], help="OpenAI model to use.")
    llm_group.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"], help="LLM temperature for creativity (0.0–1.0).")

    # --- Execution & Performance ---
    exec_group.add_argument("-d", "--dry-run", action="store_true", help="Generate .tex file but skip PDF compilation.")
    exec_group.add_argument("-i", "--interactive", action="store_true", help="Enable interactive review before generation.")
    exec_group.add_argument("--parallel", action="store_true", help="Enable parallel processing for LLM rewrites.")
    exec_group.add_argument("--no-cache", action="store_true", help="Disable all caching for this run.")
    exec_group.add_argument("--cache-clear", action="store_true", help="Clear all cached results before execution.")
    exec_group.add_argument("-v", "--verbose", action="store_true", help="Enable verbose INFO logging.")
    exec_group.add_argument("--debug", action="store_true", help="Enable DEBUG logging for granular detail.")

    return parser.parse_args()


def setup_logging(verbose: bool, debug: bool):
    """Configure logging format and level."""
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    # A cleaner log format
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    # Use force=True to reconfigure logging if it's already been set by a library
    logging.basicConfig(level=level, format=log_format, datefmt='%H:%M:%S', force=True)

    # Suppress overly verbose logs from underlying libraries
    for noisy_lib in ["urllib3", "requests", "httpx", "h11"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

# ————— Enhanced Caching System —————

class CacheManager:
    """
    File-based cache with TTL for expensive computations (e.g., LLM calls).
    Handles key hashing, TTL expiration, and robust file I/O.
    """
    def __init__(self, cache_dir: Path = Path(".cache"), ttl: int = 3600, enabled: bool = True):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Cache enabled at {self.cache_dir} with TTL={self.ttl}s")
        else:
            logging.info("Caching is disabled.")

    def _get_cache_path(self, key: str) -> Path:
        """Generates a safe, hashed path for a given cache key."""
        hashed_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{hashed_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if it's enabled and the value is still valid."""
        if not self.enabled:
            return None

        path = self._get_cache_path(key)
        if not path.exists():
            return None

        try:
            # More concise way to read and parse JSON
            data = json.loads(path.read_text("utf-8"))

            if time.time() - data.get("timestamp", 0) > self.ttl:
                logging.info(f"Cache expired for key: {key[:60]}...")
                path.unlink(missing_ok=True)
                return None

            # More concise logging
            logging.debug(f"Cache hit for key: {key[:60]}...")
            return data.get("value")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logging.warning(f"Error reading cache file {path}: {e}. Discarding.")
            path.unlink(missing_ok=True) # If the file is corrupt, delete it.
            return None

    def set(self, key: str, value: Any):
        """Store a value in the cache if caching is enabled."""
        if not self.enabled:
            return

        path = self._get_cache_path(key)
        try:
            payload = {"timestamp": time.time(), "value": value}
            # More concise way to write JSON to a file
            path.write_text(json.dumps(payload, indent=2), "utf-8")
            logging.debug(f"Cached result for key: {key[:60]}...")
        except (TypeError, IOError) as e:
            # Handle cases where the value is not JSON-serializable
            logging.warning(f"Failed to write cache for key {key[:60]}...: {e}")

    def clear(self):
        """Removes all files from the cache directory."""
        if not self.enabled or not self.cache_dir.exists():
            return
        
        removed_count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                removed_count += 1
            except OSError as e:
                logging.warning(f"Failed to delete cache file {f}: {e}")
        
        if removed_count > 0:
            logging.info(f"🧹 Cleared {removed_count} cached files.")

# ————— Enhanced Text Processing —————
# ————— Enhanced Text Processing —————

class JobDescriptionAnalyzer:
    """
    Robustly parses job descriptions using an LLM with a statistical fallback.
    Includes caching for the LLM analysis to improve performance on repeated runs.
    """
    def __init__(self, llm: Optional[ChatOpenAI] = None, cache_manager: Optional[CacheManager] = None):
        self.llm = llm
        self.cache = cache_manager
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

    def analyze_job_description(self, text: str) -> JobAnalysis:
        """
        Produce a structured analysis of a job description.
        Prioritizes a cached or fresh LLM analysis, falling back to statistical methods.
        """
        # Add the raw text to the analysis object
        if self.llm and self.cache:
            # Use a hash of the job description text as the cache key
            cache_key = f"job_analysis_{hashlib.md5(text.encode()).hexdigest()}"
            cached_analysis = self.cache.get(cache_key)
            if cached_analysis:
                logging.info("✅ Found cached job description analysis.")
                return JobAnalysis(**cached_analysis)

        if self.llm:
            try:
                logging.info("🧠 Analyzing job description with LLM...")
                llm_analysis = self._get_llm_job_analysis(text)
                llm_analysis.raw_text = text # Store raw text

                # Supplement with statistical keywords for broader coverage
                statistical_keywords = self._extract_statistical_keywords(text)
                llm_analysis.keywords.extend(k for k in statistical_keywords if k not in llm_analysis.keywords)

                if self.cache:
                    # Cache the full analysis object
                    self.cache.set(cache_key, asdict(llm_analysis))
                logging.info("✅ LLM analysis complete.")
                return llm_analysis
            except Exception as e:
                logging.warning(f"LLM-based job analysis failed: {e}. Falling back to statistical methods.")

        # Fallback to purely statistical analysis
        logging.info("⚙️ Using statistical methods for job description analysis.")
        return self._get_statistical_job_analysis(text)

    def _get_llm_job_analysis(self, text: str) -> JobAnalysis:
        """Uses an LLM to extract a detailed profile of the ideal candidate."""
        # This prompt is already well-tuned for SWE/ML roles.
        system_prompt = """You are an expert technical recruiter specializing in Software and Machine Learning Engineer roles at FAANG-level companies. Your task is to dissect a job description and extract the absolute most critical information for a candidate.
Analyze the provided job description and return a single, minified JSON object with the following keys:
- "job_title": The most accurate job title.
- "seniority_level": The seniority, e.g., "junior", "mid", "senior", "principal", "staff".
- "required_skills": A list of NON-NEGOTIABLE technologies, frameworks, and languages.
- "preferred_skills": A list of skills that are a "strong plus" or "nice to have".
- "keywords": A list of 10-15 core technical concepts and engineering responsibilities (e.g., "distributed systems", "api design", "large-scale systems"). AVOID generic business terms.
- "company": The name of the company, if found."""
        human_prompt = f"Job Description Text:\n---\n{text}\n---\nRespond with only the JSON object."
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = self.llm.invoke(messages, response_format={"type": "json_object"}).content.strip()
        data = json.loads(response)
        return JobAnalysis(**data)

    def _get_statistical_job_analysis(self, text: str) -> JobAnalysis:
        """Generates a JobAnalysis object using only statistical/regex methods."""
        keywords = self._extract_statistical_keywords(text)
        required, preferred = self._categorize_skills_by_patterns(text)
        return JobAnalysis(
            keywords=keywords,
            required_skills=required,
            preferred_skills=preferred,
            job_title=self._extract_job_title(text.splitlines()[:5]),
            seniority_level=self._determine_seniority(text),
            raw_text=text
        )

    def _extract_statistical_keywords(self, text: str, max_keywords: int = 25) -> List[str]:
        """Combines multiple statistical methods to extract keywords."""
        # Consolidating all keyword extraction logic here
        freq_keywords = self._extract_frequent_keywords(text, max_keywords)
        entity_keywords = self._extract_entity_keywords(text) if self.nlp else []
        pattern_keywords = self._extract_skill_patterns(text)
        
        all_keywords = freq_keywords + entity_keywords + pattern_keywords
        
        # More Pythonic way to deduplicate while preserving order
        return list(dict.fromkeys(kw.strip() for kw in all_keywords if len(kw.strip()) > 2))[:max_keywords]

    def _extract_frequent_keywords(self, text: str, max_keywords: int) -> List[str]:
        """Extracts the most frequent non-stopword terms from the text."""
        if not NLTK_AVAILABLE: return []
        try:
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
        except LookupError:
            logging.info("NLTK data (stopwords, punkt) not found. Downloading...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
        
        custom_stops = {'experience', 'work', 'team', 'company', 'role', 'opportunity', 'position', 'candidate', 'join', 'years', 'ability', 'skills', 'strong', 'excellent', 'good', 'knowledge', 'responsibilities', 'requirements', 'qualifications', 'plus', 'etc', 'including', 'degree'}
        stop_words.update(custom_stops)
        
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        return [word for word, count in Counter(filtered_words).most_common(max_keywords)]

    def _extract_entity_keywords(self, text: str) -> List[str]:
        """Extracts named entities and noun chunks using spaCy."""
        if not self.nlp: return []
        try:
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in {'ORG', 'PRODUCT', 'LANGUAGE'}]
            noun_chunks = [chunk.text for chunk in doc.noun_chunks if 3 < len(chunk.text) < 25]
            return list(set(entities + noun_chunks))
        except Exception as e:
            logging.warning(f"spaCy entity extraction failed: {e}")
            return []

    def _extract_skill_patterns(self, text: str) -> List[str]:
        """Extracts common technologies using regex patterns."""
        patterns = [
            r'\b(?:Python|Java|C\+\+|C#|Go|Rust|TypeScript|JavaScript|Swift|Kotlin|SQL)\b',
            r'\b(?:React|Vue|Angular|Node\.js|Express|Flask|Django|Spring)\b',
            r'\b(?:AWS|GCP|Azure|Docker|Kubernetes|Terraform|Ansible)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|BigQuery|Snowflake)\b',
            r'\b(?:Git|CI/CD|Jira)\b',
            r'\b(?:TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|LangChain)\b',
        ]
        return list(set(match.strip() for p in patterns for match in re.findall(p, text, re.IGNORECASE)))

    def _extract_job_title(self, lines: List[str]) -> str:
        """Heuristic to find the job title from the first few lines."""
        for line in lines:
            line = line.strip()
            if line and not any(line.lower().startswith(p) for p in ['job', 'role', 'position']):
                # Remove parenthetical info and company names often separated by a hyphen
                return re.sub(r'\s*\(.+\)\s*|\s+-\s+.*', '', line).strip()
        return "Software Engineer"

    def _categorize_skills_by_patterns(self, text: str) -> Tuple[List[str], List[str]]:
        """Uses regex to find 'required' and 'preferred' skill sections."""
        text_lower = text.lower()
        req_patterns = [r'(?:must have|required|you (?:need|should) have|minimum qualifications)[\s\S]{0,500}']
        pref_patterns = [r'(?:nice to have|preferred|bonus|optional|preferred qualifications)[\s\S]{0,500}']
        
        def extract_from_text(txt):
            # A more robust regex for splitting skills
            return list(set(s.strip() for s in re.split(r'[•\n,;/\-\u2022]| and | or ', txt) if 2 < len(s.strip()) < 30))

        required = [s for p in req_patterns for m in re.findall(p, text_lower) for s in extract_from_text(m)]
        preferred = [s for p in pref_patterns for m in re.findall(p, text_lower) for s in extract_from_text(m)]
        return list(set(required)), list(set(preferred))

    def _determine_seniority(self, text: str) -> str:
        """Determines seniority level based on keywords."""
        text_lower = text.lower()
        if re.search(r'\b(senior|lead|staff|principal|architect)\b', text_lower): return "senior"
        if re.search(r'\b(junior|entry|graduate|new grad|associate)\b', text_lower): return "junior"
        return "mid"
    
# ————— Config Loader and Block Manager —————

# ————— Config Loader and Block Manager —————

class ConfigLoader:
    """
    Loads and validates all user-defined resume content from a single YAML configuration file.
    """
    def __init__(self, config_path: Path):
        if not config_path.exists():
            logging.error(f"❌ Configuration file not found at: {config_path}")
            sys.exit(1)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {} # Ensure data is a dict, not None for empty files

            # Load data with fallbacks
            self.personal_info: Dict[str, str] = data.get('personal_info', {})
            self.skills: List[str] = data.get('skills', [])
            
            # Use a list comprehension for more concise block parsing
            self.blocks: List[ResumeBlock] = [
                self._parse_block(entry) for entry in data.get('blocks', [])
            ]
            
            # NEW: Validate the loaded data
            self._validate()
            
            logging.info(f"✅ Loaded and validated {len(self.blocks)} blocks, {len(self.skills)} skills, and personal info.")

        except yaml.YAMLError as e:
            logging.error(f"❌ Error parsing YAML file {config_path}: {e}")
            sys.exit(1)
        except (ValueError, TypeError) as e:
            logging.error(f"❌ Invalid data structure in '{config_path}': {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"❌ Failed to load or process config from {config_path}: {e}")
            sys.exit(1)

    def _parse_block(self, entry: Dict) -> Optional[ResumeBlock]:
        """Parses a single dictionary from YAML into a ResumeBlock object."""
        try:
            # NEW: Convert the 'section' string from YAML to our ResumeSection Enum
            section_str = entry.get('section', 'Experience')
            entry['section'] = ResumeSection(section_str)
            
            return ResumeBlock(**entry)
        except ValueError as e:
            # This catches errors if the section string is invalid (e.g., "Projecs")
            heading = entry.get('heading', 'Unknown Block')
            logging.warning(f"⚠️ Invalid section '{section_str}' for block '{heading}'. Skipping. Error: {e}")
            return None
        except TypeError as e:
            # This catches errors if keys in YAML don't match ResumeBlock fields
            heading = entry.get('heading', 'Unknown Block')
            logging.warning(f"⚠️ Mismatched fields for block '{heading}'. Skipping. Error: {e}")
            return None

    def _validate(self):
        """Performs basic type validation on the loaded configuration data."""
        if not isinstance(self.personal_info, dict):
            raise TypeError("YAML key 'personal_info' must be a dictionary.")
        if not isinstance(self.skills, list) or not all(isinstance(s, str) for s in self.skills):
            raise TypeError("YAML key 'skills' must be a list of strings.")
        if not isinstance(self.blocks, list):
            raise TypeError("YAML key 'blocks' must be a list of dictionaries.")
        
        # Filter out any blocks that failed to parse
        self.blocks = [b for b in self.blocks if b is not None]
        if not self.blocks:
            logging.warning("⚠️ No valid resume blocks were loaded after parsing.")

class ResumeBlockManager:
    """
    Scores all resume blocks and selects the most relevant ones for each section
    based on the generation configuration.
    """
    def __init__(self, all_blocks: List[ResumeBlock]):
        self.all_blocks = all_blocks

    def select_relevant_blocks(self, job_analysis: JobAnalysis, config: GenerationConfig) -> List[ResumeBlock]:
        """
        Scores all blocks and then selects the top N for each relevant section
        (Experience, Projects) based on the configuration.
        """
        if not self.all_blocks:
            logging.warning("No blocks to select from. The resume will be empty.")
            return []

        # --- 1. Score all blocks first ---
        for block in self.all_blocks:
            block.relevance_score = self._calculate_relevance_score(block, job_analysis)

        # --- 2. Separate blocks by section ---
        blocks_by_section = defaultdict(list)
        for block in self.all_blocks:
            blocks_by_section[block.section].append(block)

        # --- 3. Select top N from each section based on config ---
        selected_blocks = []
        
        # Select top Experience blocks
        experience_blocks = sorted(blocks_by_section[ResumeSection.EXPERIENCE], key=lambda b: (b.relevance_score, b.priority), reverse=True)
        selected_experience = experience_blocks[:config.max_experience_blocks]
        selected_blocks.extend(selected_experience)
        logging.info(f"🔍 Selected {len(selected_experience)}/{len(experience_blocks)} 'Experience' blocks.")
        if self.all_blocks and not selected_experience:
             logging.warning("No 'Experience' blocks were selected.")


        # Select top Projects blocks
        project_blocks = sorted(blocks_by_section[ResumeSection.PROJECTS], key=lambda b: (b.relevance_score, b.priority), reverse=True)
        selected_projects = project_blocks[:config.max_project_blocks]
        selected_blocks.extend(selected_projects)
        logging.info(f"🔍 Selected {len(selected_projects)}/{len(project_blocks)} 'Projects' blocks.")

        # You can add other sections here if needed in the future (e.g., Education)
        # education_blocks = blocks_by_section[ResumeSection.EDUCATION]
        # selected_blocks.extend(education_blocks)

        logging.info(f"Total blocks selected: {len(selected_blocks)}")
        return selected_blocks

    def _calculate_relevance_score(self, block: ResumeBlock, job: JobAnalysis) -> float:
        """
        Calculates an advanced relevance score using the job analysis.
        This version is specialized for SWE/ML roles.
        """
        # Use the new .full_text property for cleaner code
        block_text = block.full_text
        
        # --- Create a set of all known hard skills for bonus scoring ---
        all_hard_skills = {skill.lower() for skill_set in _SKILL_CATEGORIES.values() for skill in skill_set}
        all_hard_skills.update(alias.lower() for alias in _SKILL_ALIASES.values())

        # --- 1. Semantic Similarity Score (Weight: 40%) ---
        job_text_core = " ".join(job.keywords + job.required_skills)
        if not block_text.strip() or not job_text_core.strip(): return 0.0
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([block_text, job_text_core])
            semantic_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 40.0
        except ValueError:
            semantic_score = 0.0

        # --- 2. Explicit Technical Skill Score (Weight: 50%) ---
        jd_hard_skills = {s.lower() for s in (job.required_skills + job.keywords) if s.lower() in all_hard_skills}
        if not jd_hard_skills:
            tech_keyword_score = 0.0
        else:
            hits = sum(1 for skill in jd_hard_skills if skill in block_text)
            tech_keyword_score = (hits / len(jd_hard_skills)) * 50.0

        # --- 3. Priority & Impact Score (Weight: 10% + Bonus) ---
        priority_score = float(block.priority)
        quantifiable_matches = re.findall(r'\b\d[\d,.]*[m|b|k|x|%]?\b', block_text)
        impact_bonus = min(len(quantifiable_matches) * 2.5, 5.0)
        
        final_score = semantic_score + tech_keyword_score + priority_score + impact_bonus
        
        # Use a more concise logging string
        logging.debug(
            f"Scoring '{block.heading[:40]}...': "
            f"Semantic={semantic_score:.1f}, TechSkills={tech_keyword_score:.1f}, "
            f"Priority={priority_score:.1f}, Impact={impact_bonus:.1f} -> TOTAL={final_score:.1f}"
        )
        return round(final_score, 2)
    
# BulletRewriter (refactored)
# -------------------------------------------------
# ————— Bullet Rewriting —————

class BulletRewriter:
    """
    Rewrites resume bullet-points with an LLM, using parallel processing,
    and applies formatting rules like auto-bolding.
    """
    def __init__(self, config: GenerationConfig, cache_manager: CacheManager, llm: Optional[ChatOpenAI] = None):
        self.config = config
        self.cache = cache_manager
        self.llm = llm
        # Pre-compile the set of base terms for auto-bolding
        self._base_terms = {v.lower() for v in _SKILL_ALIASES.values()}
        self._base_terms.update(k.lower() for k in _SKILL_CATEGORIES.keys())

    def rewrite_blocks(self, blocks: List[ResumeBlock], job: JobAnalysis, parallel: bool = False) -> List[ResumeBlock]:
        """Rewrites bullets for a list of blocks, using parallel processing if enabled."""
        if not self.llm:
            logging.warning("⚠️  LLM unavailable – bullets will not be rewritten.")
            return blocks

        # Combine base terms with job-specific keywords for a comprehensive bolding list
        self._bold_terms = self._base_terms.union({kw.lower() for kw in job.keywords})

        # --- 1. Gather all bullets that need to be rewritten ---
        tasks = []
        blocks_to_process_map = {} # Maps heading to block object for easy reassembly
        
        for block in blocks:
            # The Skills section is generated from the master list, not rewritten.
            if block.section == ResumeSection.SKILLS:
                continue

            cache_key = self._make_cache_key(block, job)
            if self.config.use_cache and self.cache.get(cache_key):
                logging.info(f"✅ Cache hit for block: {block.heading[:40]}...")
                block.bullets = self.cache.get(cache_key)
                continue
            
            blocks_to_process_map[block.heading] = block
            for i, bullet_text in enumerate(block.bullets[:self.config.max_bullets_per_block]):
                tasks.append((block.heading, i, bullet_text, block, job))

        if not tasks:
            logging.info("All relevant blocks are already cached. No rewriting needed.")
            return blocks

        # --- 2. Execute rewriting tasks ---
        results_map = {}
        with ThreadPoolExecutor(max_workers=8 if parallel else 1) as executor:
            future_to_task = {executor.submit(self._rewrite_via_llm, t[2], t[3], t[4]): t for t in tasks}
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="✏️ Rewriting Bullets"):
                heading, idx, original_bullet, _, _ = future_to_task[future]
                try:
                    results_map[(heading, idx)] = future.result()
                except Exception as e:
                    logging.error(f"Error rewriting bullet for '{heading}': {e}")
                    results_map[(heading, idx)] = original_bullet # Fallback to original text

        # --- 3. Reassemble and format the rewritten bullets ---
        for heading, block in blocks_to_process_map.items():
            # Apply formatting to the rewritten bullets
            new_bullets = [
                self._format_bullet(results_map.get((heading, i), b))
                for i, b in enumerate(block.bullets[:self.config.max_bullets_per_block])
            ]
            block.bullets = new_bullets
            
            if self.config.use_cache:
                self.cache.set(self._make_cache_key(block, job), block.bullets)
        return blocks

    def _rewrite_via_llm(self, bullet: str, block: ResumeBlock, job: JobAnalysis) -> str:
        """Invokes the LLM to rewrite a single bullet point."""
        messages = [SystemMessage(content=self._system_prompt()), HumanMessage(content=self._human_prompt(bullet, block, job))]
        return self.llm.invoke(messages).content.strip()

    def _make_cache_key(self, block: ResumeBlock, job: JobAnalysis) -> str:
        """Creates a unique cache key based on content and LLM settings."""
        # Simplified to remove style/focus which are now baked into the main prompt
        key_parts = [
            self.config.model, 
            str(self.config.temperature),
            block.heading, 
            "|".join(sorted(block.bullets)), 
            ",".join(sorted(job.keywords))
        ]
        return hashlib.md5("_".join(key_parts).encode("utf-8")).hexdigest()

    def _system_prompt(self) -> str:
        """The specialized system prompt for SWE/ML roles."""
        # This prompt is now static and specialized, removing the need for style/focus flags.
        return """You are a top-tier resume writer for Senior Software Engineers. Your goal is to rewrite a bullet point to highlight strong engineering principles, system impact, and clean design. The target audience is an engineering manager or architect.

CRITICAL FORMAT RULES:
• Start with a strong, active verb that describes an engineering action (e.g., Developed, Architected, Refactored, Deployed, Maintained, Scaled).
• Quantify impact whenever possible (e.g., '...improving API response time by 30%', '...supporting 500k active users', '...reducing infrastructure costs by 20%'). If no metric exists, describe the scale or complexity of the system.
• Mention the key technologies used to build the solution. Bold them using LaTeX: \\textbf{Go}.
• Focus on the 'how' and 'why' of the technical achievement, not just the 'what'.
• NEVER use personal pronouns (I, we, my).
• DO NOT end with a period.
• Return ONLY the single, rewritten bullet point."""

    def _human_prompt(self, bullet: str, block: ResumeBlock, job: JobAnalysis) -> str:
        """Creates the user-facing prompt with context for the LLM."""
        return f"Job Title: {job.job_title}\nTop Keywords: {', '.join(job.keywords[:10])}\nRequired Skills: {', '.join(job.required_skills[:5])}\nResume Section: {block.section.value} — {block.heading}\nOriginal Bullet: {bullet}\n\nRewrite the bullet following the FORMAT RULES."

    def _format_bullet(self, bullet: str) -> str:
        """Applies final cleaning, capitalization, and auto-bolding."""
        if not bullet: return ""
        # Clean up common LLM artifacts like leading list markers
        bullet = re.sub(r'^[•·‣▪▫⁃\-–]+\s*', '', bullet.strip())
        if not bullet: return ""
        
        bullet = bullet[0].upper() + bullet[1:]
        bullet = bullet.rstrip('.')
        bullet = self._auto_bold(bullet)

        # Truncate long bullets gracefully
        if len(bullet) > 250:
            cut_point = bullet.rfind(' ', 0, 250)
            bullet = bullet[:cut_point] + '…' if cut_point != -1 else bullet[:250] + '…'
        return bullet

    def _auto_bold(self, text: str) -> str:
        """
        Bolds known technical terms in the text without using complex regex look-behinds.
        """
        # Final cleanup for any accidental double-bolding from the LLM
        text = re.sub(r'\\textbf\{\s*\\textbf\{([^}]+)\}\s*\}', r'\\textbf{\1}', text, flags=re.IGNORECASE)

        for term in sorted(self._bold_terms, key=len, reverse=True):
            # This regex finds the term as a whole word, case-insensitively
            pattern = re.compile(r'\b({})\b'.format(re.escape(term)), re.IGNORECASE)
            
            # Use a function as the replacement to check context
            def replacer(match):
                # Check if the match is already inside a \textbf{...} block
                # This is a simple heuristic: check for an unclosed brace before the match
                pre_text = text[:match.start()]
                if pre_text.count('{') > pre_text.count('}'):
                    return match.group(0) # It's already bolded, return the original match
                else:
                    return r'\textbf{' + match.group(0) + '}' # It's safe to bold
            
            text = pattern.sub(replacer, text)
        return text
    
# ————————————————
#  LaTeXGenerator (refactored)
# ————————————————

# ————— LaTeX Generation & PDF Compilation —————

class LaTeXGenerator:
    """
    Renders resume content into a LaTeX string using a Jinja2 template,
    with robust escaping and a fallback generator.
    """
    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False, # We handle escaping manually
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined
        )
        self.env.filters['latex_escape'] = self._latex_escape

    def generate_resume(
        self,
        personal: Dict[str, str],
        skills_by_category: Dict[str, List[str]],
        experience: List[ResumeBlock],
        projects: List[ResumeBlock]
    ) -> str:
        """Renders the resume from a Jinja2 template."""
        template_context = {
            **personal,
            "skills_by_category": skills_by_category,
            "experience": experience,
            "projects": projects,
        }
        try:
            template = self.env.get_template("resume_template.tex")
            return template.render(template_context)
        except Exception as e:
            logging.error(f"🛑 Template rendering failed: {e}. Generating fallback LaTeX.")
            # We don't need _post_process_latex here as the fallback is simple and clean
            return self._generate_fallback_latex(personal, experience + projects)

    def _simple_escape(self, text: str) -> str:
        """A helper that escapes special LaTeX characters in a plain string."""
        specials = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
        for char, esc in specials.items():
            # Use regex to avoid double-escaping already escaped characters
            text = re.sub(rf'(?<!\\){re.escape(char)}', esc.replace('\\', '\\\\'), text)
        return text
    
    def _latex_escape(self, text: str) -> str:
        """
        Robustly escapes a string for LaTeX, preserving \\textbf commands
        while correctly escaping their content.
        """
        if not isinstance(text, str):
            return text

        # A more concise list comprehension approach to the escape logic
        parts = re.split(r'(\\textbf{)(.*?)(})', text)
        escaped_parts = []
        for i, part in enumerate(parts):
            # Parts at index 0, 4, 8, ... are the text outside the commands
            if i % 4 == 0:
                escaped_parts.append(self._simple_escape(part))
            # The other parts are the command itself (\textbf{), its content, and the closing brace }
            # which we just append as is, since the content will be handled in the next iteration.
            # We only need to reconstruct the `\textbf{content}` part, so we check for the command start.
            elif parts[i-1] == r'\textbf{':
                 escaped_parts.append(r'\textbf{' + self._simple_escape(part) + '}')

        return "".join(escaped_parts)
    
    def _generate_fallback_latex(self, personal: Dict, blocks: List[ResumeBlock]) -> str:
        """Generates a minimal, safe LaTeX document in case of template failure."""
        name = self._simple_escape(personal.get('name', 'Your Name'))
        email = self._simple_escape(personal.get('email', 'your.email@example.com'))
        phone = self._simple_escape(personal.get('phone', '(555) 123-4567'))

        header = f"""\\documentclass[letterpaper,11pt]{{article}}
\\usepackage[left=0.75in,top=0.6in,right=0.75in,bottom=0.6in]{{geometry}}
\\usepackage{{titlesec}}
\\usepackage{{enumitem}}
\\setlist[itemize]{{noitemsep, topsep=0pt}}
\\begin{{document}}
\\begin{{center}}
{{\\Large \\textbf{{{name}}}}}\\\\{email} | {phone}
\\end{{center}}
"""
        
        body = []
        # Group blocks by section
        sections = defaultdict(list)
        for b in blocks:
            sections[b.section.value].append(b)

        for section_name, section_blocks in sections.items():
            body.append(f'\\section*{{{self._simple_escape(section_name)}}}')
            for blk in section_blocks:
                heading = self._simple_escape(blk.heading)
                date = self._simple_escape(blk.date)
                body.append(f'\\textbf{{{heading}}} \\hfill {date}')
                body.append('\\begin{itemize}')
                body.extend(f'\\item {self._simple_escape(b)}' for b in blk.bullets)
                body.append('\\end{itemize}\\vspace{{5pt}}')
        
        return '\n'.join([header] + body + ['\\end{document}'])
    
# ---------------- PDF Compilation ---------------- #

# ————— LaTeX Generation & PDF Compilation —————

# NEW: A dedicated dataclass for compilation results improves clarity.
@dataclass
class CompilationResult:
    """Stores the outcome of a LaTeX compilation."""
    pdf_path: Optional[Path] = None
    log: Optional[str] = None
    success: bool = False
    
    @property
    def log_path(self) -> Optional[Path]:
        return self.pdf_path.with_suffix(".log") if self.pdf_path else None

class PDFCompiler:
    """
    Compiles a LaTeX string to a PDF file with robust error handling,
    including a two-pass run for complex documents.
    """
    def __init__(self, output_dir: Path, compiler: str = "xelatex", timeout: int = 60):
        self.output_dir = output_dir
        self.compiler = compiler
        self.timeout = timeout
        if not shutil.which(self.compiler):
            raise FileNotFoundError(f"✘ LaTeX engine '{self.compiler}' not found in PATH.")

    def compile_latex(self, latex_content: str, output_name: str, keep_logs: bool = False) -> CompilationResult:
        """Writes and compiles a LaTeX string, returning a result object."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tex_file = self.output_dir / f"{output_name}.tex"
        pdf_file = self.output_dir / f"{output_name}.pdf"
        
        try:
            tex_file.write_text(latex_content, encoding="utf-8")
            logging.info(f"➜ LaTeX written to: {tex_file}")
        except IOError as e:
            logging.error(f"❌ Failed to write .tex file: {e}")
            return CompilationResult(success=False)

        if warnings := self._validate_latex(latex_content):
            logging.warning(f"⚠ LaTeX validation issues found: {', '.join(warnings)}")

        # --- Run compilation (up to two passes) ---
        first_pass_result = self._run_compiler(tex_file)
        if not first_pass_result.success:
            self._parse_and_log_errors(first_pass_result)
            return first_pass_result
        
        # A second pass is often needed for table of contents, references, etc.
        # It's good practice and low cost.
        logging.info("Running a second compilation pass for final document assembly...")
        final_result = self._run_compiler(tex_file)

        if not final_result.success:
            self._parse_and_log_errors(final_result)
            return final_result

        if not pdf_file.exists():
            logging.error("✘ PDF file not found after successful compilation.")
            return CompilationResult(success=False, log=final_result.log)
        
        final_result.pdf_path = pdf_file
        logging.info(f"✓ PDF compiled successfully → {pdf_file}")

        if not keep_logs:
            self._cleanup_aux_files(tex_file)
        
        return final_result

    def _run_compiler(self, tex_file: Path) -> CompilationResult:
        """Executes a single pass of the LaTeX compiler."""
        cmd = [
            self.compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={self.output_dir}",
            str(tex_file),
        ]
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if process.returncode == 0:
                return CompilationResult(success=True, log=process.stdout)
            else:
                return CompilationResult(success=False, log=process.stdout)
        except subprocess.TimeoutExpired:
            logging.error("✘ LaTeX compilation timed-out.")
            return CompilationResult(success=False, log="Compilation timed out.")

    def _validate_latex(self, src: str) -> List[str]:
        """Performs lightweight static checks for common LaTeX errors."""
        issues = []
        if src.count("{") != src.count("}"): issues.append("Unbalanced braces {}")
        # A more robust check for unescaped special characters
        for char in "&%$#_":
            # Find a character that is NOT preceded by a backslash
            if re.search(rf"[^\\]{re.escape(char)}", src):
                issues.append(f"Possibly unescaped '{char}' character")
        for token in ["\\documentclass", "\\begin{document}", "\\end{document}"]:
            if token not in src: issues.append(f"Missing '{token}'")
        return issues

    def _parse_and_log_errors(self, result: CompilationResult):
        """Extracts and logs the first few LaTeX errors from the compiler output."""
        logging.error(f"✘ {self.compiler} compilation failed. See log for details.")
        error_lines = [line.strip() for line in result.log.splitlines() if line.startswith("!")]
        if error_lines:
            logging.error("LaTeX errors found:")
            for line in error_lines[:5]:
                logging.error(f"  > {line}")

    def _cleanup_aux_files(self, tex_file: Path):
        """Removes auxiliary files produced by the LaTeX compiler."""
        for ext in {".aux", ".log", ".out", ".toc", ".fdb_latexmk", ".fls"}:
            aux_file = tex_file.with_suffix(ext)
            if aux_file.exists():
                aux_file.unlink(missing_ok=True)

# ————— Interactive Mode —————

class InteractiveSession:
    """
    Manages an interactive session for reviewing, editing, and confirming
    the resume blocks selected for generation.
    """
    def __init__(self, job_analysis: JobAnalysis, selected_blocks: List[ResumeBlock], all_blocks: List[ResumeBlock]):
        self.job_analysis = job_analysis
        self.selected_blocks = selected_blocks
        self.all_blocks = all_blocks # Keep a reference to all blocks for swapping

    def run(self) -> List[ResumeBlock]:
        """Starts and manages the interactive command loop."""
        print(f"\n{'='*60}\n🧠 INTERACTIVE RESUME REVIEW\n{'='*60}")
        self._display_job_analysis()

        while True:
            self._display_selected_blocks()
            
            command = input("\nEnter command (p <#> to preview, r <#> to remove, s <#> <#> to swap, a to view all, c to continue, q to quit): ").strip().lower()
            parts = command.split()
            cmd = parts[0] if parts else ''

            if cmd == 'p': self._preview_block_cmd(parts)
            elif cmd == 'r': self._remove_block_cmd(parts)
            elif cmd == 's': self._swap_block_cmd(parts)
            elif cmd == 'a': self._preview_all_blocks()
            elif cmd == 'c':
                print("✅ Continuing with resume generation...\n")
                return self.selected_blocks
            elif cmd == 'q':
                if input("Are you sure you want to quit? (y/n): ").lower() == 'y':
                    sys.exit(0)
            else:
                print("❌ Invalid command. Please try again.")

    def _display_job_analysis(self):
        print(f"\n📌 Job Analysis:")
        print(f"  Title: {self.job_analysis.job_title} ({self.job_analysis.seniority_level})")
        print(f"  Keywords: {', '.join(self.job_analysis.keywords[:8])}...")

    def _display_selected_blocks(self):
        print(f"\n📂 Selected Resume Blocks ({len(self.selected_blocks)}):")
        for i, block in enumerate(self.selected_blocks, 1):
            print(f"  {i}. {block.heading[:70]}... (Score: {block.relevance_score:.1f})")

    def _display_block_details(self, block: ResumeBlock):
        print(f"\n--- Details for: {block.heading} ---")
        print(f"  Section: {block.section.value:<12} | Date: {block.date:<20} | Score: {block.relevance_score:.2f}")
        print("  Bullets:")
        for bullet in block.bullets:
            print(f"    • {bullet}")
        print("-" * (len(block.heading) + 18))

    def _preview_block_cmd(self, parts: List[str]):
        if len(parts) != 2:
            print("Usage: p <number>")
            return
        try:
            idx = int(parts[1]) - 1
            if 0 <= idx < len(self.selected_blocks):
                self._display_block_details(self.selected_blocks[idx])
            else:
                print("❌ Invalid block number.")
        except ValueError:
            print("❌ Invalid number.")

    def _preview_all_blocks(self):
        for block in self.selected_blocks:
            self._display_block_details(block)

    def _remove_block_cmd(self, parts: List[str]):
        if len(parts) != 2:
            print("Usage: r <number>")
            return
        try:
            idx = int(parts[1]) - 1
            if 0 <= idx < len(self.selected_blocks):
                removed = self.selected_blocks.pop(idx)
                print(f"🗑️  Removed: {removed.heading}")
            else:
                print("❌ Invalid block number.")
        except ValueError:
            print("❌ Invalid number.")
            
    def _swap_block_cmd(self, parts: List[str]):
        """NEW: Allows swapping a selected block with an unselected one."""
        if len(parts) != 3:
            print("Usage: s <selected_num> <unselected_num>")
            return
        try:
            selected_idx = int(parts[1]) - 1
            unselected_idx = int(parts[2]) - 1

            unselected_pool = [b for b in self.all_blocks if b not in self.selected_blocks]
            unselected_pool.sort(key=lambda b: (b.relevance_score, b.priority), reverse=True)

            if not (0 <= selected_idx < len(self.selected_blocks)):
                print("❌ Invalid number for the selected block.")
                return
            if not (0 <= unselected_idx < len(unselected_pool)):
                print("❌ Invalid number for the unselected block.")
                return

            # Perform the swap
            swapped_out = self.selected_blocks[selected_idx]
            swapped_in = unselected_pool[unselected_idx]
            self.selected_blocks[selected_idx] = swapped_in
            
            print(f"✅ Swapped '{swapped_out.heading[:30]}...' with '{swapped_in.heading[:30]}...'")
            
        except ValueError:
            print("❌ Invalid number(s).")

# ————— Skill Processing —————

# NEW: Tuned dictionaries for a high-level SWE/ML Engineer profile.
_SKILL_CATEGORIES: dict[str, set[str]] = {
    "Languages & Core": {"Python", "C++", "Java", "Go", "SQL", "Bash"},
    "ML/DL Frameworks": {"PyTorch", "TensorFlow", "JAX", "Keras", "Scikit-learn", "Hugging Face Transformers"},
    "ML Infra & MLOps": {"Kubernetes", "Docker", "MLflow", "Weights & Biases", "Kubeflow", "Ray", "Triton Inference Server"},
    "Data & Distributed Systems": {"Apache Spark", "Pandas", "NumPy", "Dask", "Kafka", "Airflow", "PostgreSQL"},
    "LLM & Vector DBs": {"LangChain", "LlamaIndex", "Weaviate", "Pinecone", "Milvus", "FAISS", "OpenAI API"},
    "Cloud & DevOps": {"AWS", "GCP", "Azure", "Terraform", "CI/CD", "GitHub Actions", "Ansible"},
    "Web & API": {"FastAPI", "gRPC", "GraphQL", "React", "Node.js"},
}

_SKILL_ALIASES: dict[str, str] = {
    # This map helps normalize and shorten skill names for display
    "C++": "C++", "Hugging Face Transformers": "Hugging Face", "Scikit-learn": "Sklearn",
    "Weights & Biases": "W&B", "Apache Spark": "Spark", "PostgreSQL": "Postgres",
    "Node.js": "Node", "Kubernetes": "K8s", "GitHub Actions": "GH Actions",
    "Amazon Web Services": "AWS", "Google Cloud Platform": "GCP", "Triton Inference Server": "Triton",
}


# NEW: Encapsulate all skill-related logic into a single class.
class SkillsManager:
    """Handles filtering, categorizing, and preparing skills for rendering."""

    def __init__(self, all_skills: List[str]):
        # Create a reverse map for efficient categorization
        self.skill_to_category: Dict[str, str] = {
            skill: category
            for category, skill_set in _SKILL_CATEGORIES.items()
            for skill in skill_set
        }
        self.all_skills = set(all_skills)

    def prepare_for_rendering(self, job_keywords: List[str], min_total: int = 20) -> Dict[str, List[str]]:
        """
        The main entry point to get the final, categorized skills for the resume.
        """
        # 1. Filter skills to find ones relevant to the job description
        relevant_skills = self._filter_relevant_skills(job_keywords)
        
        # 2. Top-up the list with other important skills if we don't meet the minimum
        if len(relevant_skills) < min_total:
            needed = min_total - len(relevant_skills)
            # Prioritize skills from our master list that weren't deemed relevant
            additional_skills = [s for s in self.all_skills if s not in relevant_skills][:needed]
            relevant_skills.update(additional_skills)

        # 3. Categorize and apply aliases to the final set of skills
        categorized = defaultdict(list)
        for skill in relevant_skills:
            category = self.skill_to_category.get(skill, "Other Frameworks")
            display_name = _SKILL_ALIASES.get(skill, skill)
            categorized[category].append(display_name)
        
        # 4. Sort and return the final dictionary
        return {k: sorted(v) for k, v in sorted(categorized.items())}

    def _filter_relevant_skills(self, job_keywords: List[str]) -> set[str]:
        """
        Efficiently finds relevant skills using set intersections.
        """
        # Create a lookup set of all parts of all job keywords
        job_keyword_parts = {part.lower() for kw in job_keywords for part in re.split(r'\W+', kw) if part}
        
        relevant = set()
        for skill in self.all_skills:
            # Check if any part of the skill name is in the job keyword parts
            skill_parts = {part.lower() for part in re.split(r'\W+', skill) if part}
            if not skill_parts.isdisjoint(job_keyword_parts):
                relevant.add(skill)
        
        return relevant

# This is the single function that will be called from the main application.
def prepare_skills_for_rendering(
    all_skills: list[str],
    job_keywords: list[str],
    min_total: int = 20
) -> dict[str, list[str]]:
    """Creates a SkillsManager and runs the preparation process."""
    manager = SkillsManager(all_skills)
    return manager.prepare_for_rendering(job_keywords, min_total)

# ————— Application Class & Main Entry Point —————

class ResumeGeneratorApp:
    """Encapsulates the entire resume generation workflow."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # Use the new .from_args() classmethod to create the config
        self.config = GenerationConfig.from_args(args)
        self.cache_manager = None
        self.llm_instance = None
        self.config_loader = None

    def setup(self):
        """Initializes logging, caching, configuration, and the LLM."""
        setup_logging(self.args.verbose, self.args.debug)
        load_dotenv()

        # Initialize the cache manager using the 'enabled' flag
        self.cache_manager = CacheManager(enabled=self.config.use_cache)
        if self.args.cache_clear:
            self.cache_manager.clear()

        # Load the main resume configuration from the YAML file
        logging.info(f"📂 Loading resume configuration from: {self.args.blocks}")
        self.config_loader = ConfigLoader(self.args.blocks)
        
        # Validate other necessary file paths
        _validate_files(self.args)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        
        if LANGCHAIN_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                self.llm_instance = ChatOpenAI(
                    model=self.config.model,
                    temperature=self.config.temperature
                )
            except Exception as e:
                logging.error(f"❌ Could not initialize LLM: {e}")
        else:
            logging.warning("LLM features are disabled (LangChain not installed or OPENAI_API_KEY not set).")

    def run(self):
        """Executes the main application logic from analysis to compilation."""
        try:
            # 1. Analysis (now with caching)
            analyzer = JobDescriptionAnalyzer(llm=self.llm_instance, cache_manager=self.cache_manager)
            jd_text = self.args.jd.read_text(encoding='utf-8')
            job_analysis = analyzer.analyze_job_description(jd_text)

            # 2. Block Selection (using the new method signature)
            block_manager = ResumeBlockManager(self.config_loader.blocks)
            selected_blocks = block_manager.select_relevant_blocks(job_analysis, self.config)
            if not selected_blocks:
                logging.error("❌ No resume blocks were selected. Check scoring or job description.")
                sys.exit(1)

            # 3. Interactive Mode (using the new class-based session)
            if self.args.interactive:
                session = InteractiveSession(job_analysis, selected_blocks, self.config_loader.blocks)
                selected_blocks = session.run()

            # 4. Bullet Rewriting
            rewriter = BulletRewriter(self.config, self.cache_manager, self.llm_instance)
            rewritten_blocks = rewriter.rewrite_blocks(selected_blocks, job_analysis, self.args.parallel)
            
            # 5. LaTeX Generation and Compilation
            self._generate_and_compile(rewritten_blocks, job_analysis)

        except KeyboardInterrupt:
            logging.info("\n🛑 Interrupted by user.")
            sys.exit(130)
        except Exception as e:
            logging.exception(f"💥 An unexpected error occurred: {e}")
            sys.exit(1)

    def _generate_and_compile(self, blocks: List[ResumeBlock], job: JobAnalysis):
        """Handles the final rendering and PDF compilation steps."""
        generator = LaTeXGenerator(self.args.template_dir)
        
        sections = defaultdict(list)
        for blk in blocks:
            sections[blk.section].append(blk)
        
        # Use the dedicated SkillsManager for this logic
        skills_manager = SkillsManager(self.config_loader.skills)
        skills_by_category = skills_manager.prepare_for_rendering(job.keywords)

        latex_content = generator.generate_resume(
            personal=self.config_loader.personal_info,
            skills_by_category=skills_by_category, 
            experience=sections.get(ResumeSection.EXPERIENCE, []),
            projects=sections.get(ResumeSection.PROJECTS, [])
        )

        tex_file = self.args.output_dir / f"{self.args.output_name}.tex"
        tex_file.write_text(latex_content, encoding='utf-8')
        logging.info(f"📄 LaTeX file saved to: {tex_file}")

        if not self.args.dry_run:
            # Pass the debug flag to the compiler to control log retention
            compiler = PDFCompiler(self.args.output_dir)
            result = compiler.compile_latex(latex_content, self.args.output_name, keep_logs=self.args.debug)
            if result.success:
                _try_open(result.pdf_path)
            else:
                sys.exit(1)
        else:
            logging.info("ℹ️ Dry run complete. No PDF compiled.")
            
# ————— Main Application —————
def main():
    """Parses args and runs the main application."""
    args = parse_args()
    app = ResumeGeneratorApp(args)
    app.setup()
    app.run()

# Keep your _try_open and _validate_files helpers, but _validate_files can be simplified.
def _try_open(pdf_file: Path):
    """Try to open the PDF using platform-specific methods."""
    try:
        if sys.platform.startswith('darwin'):
            subprocess.run(['open', str(pdf_file)], check=False)
        elif sys.platform.startswith('win'):
            subprocess.run(['start', str(pdf_file)], shell=True, check=False)
        elif sys.platform.startswith('linux'):
            subprocess.run(['xdg-open', str(pdf_file)], check=False)
    except Exception as e:
        logging.warning(f"Could not open PDF file: {e}")

def _validate_files(args) -> None:
    """Validate that required input files and directories exist."""
    # The main YAML config is validated in ConfigLoader. We check the rest.
    required_paths = {
        "Job description file": args.jd,
        "Template directory": args.template_dir,
    }
    
    for description, path in required_paths.items():
        if not path.exists():
            logging.error(f"❌ Missing required input: {description} not found at '{path}'")
            sys.exit(1)

    template_file = args.template_dir / "resume_template.tex"
    if not template_file.exists():
        logging.warning(f"⚠️  Template file not found at '{template_file}'. The script will likely fail.")

if __name__ == "__main__":
    main()

