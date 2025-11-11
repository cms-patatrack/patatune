import os

def on_page_read_source(page, config):
    if page.file.src_path == 'index.md':
        
        readme_path = './README.md'
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        styled_content = '<style> .md-content .md-typeset h1 { display: none; } </style>\n\n' + content
        
        return styled_content
            
    return None