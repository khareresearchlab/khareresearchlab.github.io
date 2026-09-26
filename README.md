# Khare Lab research website

A responsive, multi-page static website built with plain HTML, CSS and JavaScript. It is designed to publish on GitHub Pages without a build step.

## Pages

- `index.html` — introduction and overview
- `research.html` — research themes
- `group.html` — PI and group members
- `publications.html` — selected publications
- `gallery.html` — photos and field/lab gallery
- `contact.html` — contact details

## Before publishing

Search for `ADD-` and `placeholder` in the files. Replace the email address, group member names, biography details, publication entries, profile links and image placeholders. Do not publish placeholder citations or contact details.

## Preview locally

Open `index.html` in a browser, or from this folder run:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Publish with GitHub Pages

1. Create a GitHub repository named `YOUR-USERNAME.github.io` (a public repository is simplest for a first Pages site).
2. Upload the contents of this folder to the repository root, including `index.html` and the `assets` folder.
3. In the repository, open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/ (root)`, then save.
5. Wait for the Pages build to finish. Your site will appear at `https://YOUR-USERNAME.github.io/`.

If the repository has a different name, the URL will be `https://YOUR-USERNAME.github.io/REPOSITORY-NAME/`.
