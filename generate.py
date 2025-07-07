import argparse, subprocess, yaml
from jinja2 import Environment, FileSystemLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_blocks(path="blocks_dynamic.yaml"):
    return yaml.safe_load(open(path))

def pick_top(blocks, jd_text, section, top_n):
    items = [b for b in blocks if b["section"] == section]
    texts = [
        (" ".join(b.get("bullets", [])) if section=="Skills"
         else b["heading"] + " " + " ".join(b["bullets"]))
        for b in items
    ]
    vecs = TfidfVectorizer(stop_words="english").fit_transform(texts + [jd_text])
    sims = cosine_similarity(vecs[:-1], vecs[-1])
    scored = sorted(zip(items, sims.ravel()), key=lambda t: -t[1])
    return [b for b,_ in scored[:top_n]]

def build_projects_block(projects):
    lines = []
    for p in projects:
        lines.append(f"\\resumeProjectHeading{{{p['heading']}}}{{{p.get('date','')}}}")
        lines.append("\\resumeItemListStart")
        for bullet in p["bullets"]:
            lines.append(f"\\resumeItem{{{bullet}}}")
        lines.append("\\resumeItemListEnd")
    return "\n".join(lines)

def build_skills_block(skills):
    # Flatten to a simple list of skill strings
    top = [b["bullets"][0] for b in skills]

    # Define your categories & the master lists
    categories = {
        "Languages":      ["Python","Java","Kotlin","C","MATLAB","JavaScript","TypeScript"],
        "ML/DL":          ["PyTorch","TensorFlow","PointNet++","CLIP","ONNX","Optuna","Scikit-learn","LangChain"],
        "Audio/DSP":      ["TorchAudio","Pedalboard","Librosa","FFT","Real-Time Audio"],
        "Backend":        ["Node.js","Express","MongoDB","Docker","AWS","REST APIs"],
        "Frontend":       ["React","Tailwind CSS","Chakra UI","Mapbox GL JS"],
        "Tools":          ["Git","Linux","VS Code","Jupyter","Figma"]
    }

    # Build the single LaTeX item
    lines = ["\\item \\small{"]
    for cat, pool in categories.items():
        chosen = [s for s in top if s in pool]
        if chosen:
            # join with comma+space, end with \\ for linebreak
            lines.append(f"  \\textbf{{{cat}:}} {', '.join(chosen)} \\\\")
    lines.append("}")

    # wrap in resumeSubHeadingListStart/End is in the template
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default="job_description.txt")
    parser.add_argument("--projects", type=int, default=5)
    parser.add_argument("--skills",   type=int, default=8)
    args = parser.parse_args()

    jd = open(args.jd).read()
    blocks = read_blocks()
    top_projects = pick_top(blocks, jd, "Projects", args.projects)
    top_skills   = pick_top(blocks, jd, "Skills",   args.skills)

    projects_block = build_projects_block(top_projects)
    skills_block   = build_skills_block(top_skills)

    env = Environment(loader=FileSystemLoader("templates"), autoescape=False)
    tpl = env.get_template("resume_template.tex")
    rendered = tpl.render(projects_block=projects_block, skills_block=skills_block)

    with open("tailored_resume.tex", "w") as f:
        f.write(rendered)
    subprocess.run(["pdflatex","-interaction=nonstopmode","tailored_resume.tex"], check=True)

if __name__=="__main__":
    main()
