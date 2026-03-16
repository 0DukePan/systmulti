import markdown2
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def generate_pdf():
    with open('Rapport_TP5_RLE_Codec.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'header-ids'])

    css = """
    @page {
        size: A4;
        margin: 2.5cm 2cm 2.5cm 2.5cm;

    }
    h1:first-of-type {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        margin-top: 2em;
        margin-bottom: 1em;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5em;
    }
    body {
        font-family: 'Georgia', serif;
        line-height: 1.65;
        color: #2c3e50;
        font-size: 11pt;
        text-align: justify;
    }
    h1, h2, h3, h4 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1a252f;
        margin-top: 1.8em;
        margin-bottom: 0.8em;
        page-break-after: avoid;
        font-weight: 600;
    }
    h2 {
        font-size: 1.6em;
        color: #2980b9;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.3em;
    }
    h3 { font-size: 1.3em; color: #34495e; }
    h4 { font-size: 1.1em; color: #7f8c8d; font-style: italic; }
    p { margin-bottom: 1.2em; }
    strong { color: #2980b9; }
    ul, ol { margin-bottom: 1.5em; padding-left: 2em; }
    li { margin-bottom: 0.4em; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 2em 0;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 10pt;
    }
    th, td {
        border: 1px solid #ecf0f1;
        padding: 12px 15px;
        text-align: left;
    }
    th {
        background-color: #34495e;
        color: white;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    tr:nth-child(even) { background-color: #f9fbfd; }
    pre {
        background-color: #f4f6f7;
        border-left: 4px solid #3498db;
        border-radius: 0 4px 4px 0;
        padding: 1.2em;
        margin: 1.5em 0;
        overflow-x: auto;
        font-family: 'Consolas', 'Menlo', monospace;
        font-size: 9pt;
        line-height: 1.4;
    }
    code {
        font-family: 'Consolas', 'Menlo', monospace;
        background-color: #f4f6f7;
        color: #c0392b;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-size: 0.95em;
    }
    pre code { background-color: transparent; color: #333; padding: 0; font-size: 1em; }
    blockquote {
        margin: 1.5em 0;
        padding: 1em 1.5em;
        border-left: 5px solid #e74c3c;
        background-color: #fdf2f1;
        color: #555;
        font-style: italic;
    }
    a { color: #3498db; text-decoration: none; }
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(52,152,219,0.75), rgba(0,0,0,0));
        margin: 2.5em 0;
    }
    """

    html_string = f"""<!DOCTYPE html>
    <html lang="fr">
    <head><meta charset="utf-8"><title>Rapport TP5</title><style>{css}</style></head>
    <body>{html_content}</body>
    </html>"""

    font_config = FontConfiguration()
    print("Generating PDF...")
    HTML(string=html_string, base_url='.').write_pdf(
        'Rapport_TP5_RLE_Codec.pdf',
        font_config=font_config,
        presentational_hints=True
    )
    print("PDF generated: Rapport_TP5_RLE_Codec.pdf")

if __name__ == '__main__':
    generate_pdf()
