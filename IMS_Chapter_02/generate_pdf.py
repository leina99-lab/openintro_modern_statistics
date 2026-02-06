#!/usr/bin/env python3
"""
IMS Chapter 2 PDF 생성 스크립트
WeasyPrint 사용, 한글 폰트: Noto Sans CJK KR
수식은 LaTeX 대신 유니코드 사용
"""

import markdown
from weasyprint import HTML, CSS
import re
import os

def convert_latex_to_unicode(text):
    """LaTeX 수식을 유니코드로 변환"""
    replacements = {
        r'\bar{x}': 'x̄',
        r'\hat{p}': 'p̂',
        r'\mu': 'μ',
        r'\sigma': 'σ',
        r'\rho': 'ρ',
        r'\pi': 'π',
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\gamma': 'γ',
        r'\delta': 'δ',
        r'\epsilon': 'ε',
        r'\lambda': 'λ',
        r'\theta': 'θ',
        r'\rightarrow': '→',
        r'\leftarrow': '←',
        r'\neq': '≠',
        r'\leq': '≤',
        r'\geq': '≥',
        r'\times': '×',
        r'\pm': '±',
        r'\sum': 'Σ',
        r'\sqrt': '√',
        r'\infty': '∞',
        r'\\': ' ',
    }
    
    for latex, unicode_char in replacements.items():
        text = text.replace(latex, unicode_char)
    
    # $...$ 사이의 내용 처리 (단순 텍스트로)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    
    return text

def main():
    print("=" * 50)
    print("IMS Chapter 2 PDF 생성 시작")
    print("=" * 50)
    
    # 마크다운 파일 읽기
    with open('Chapter_02_Study_Design.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # LaTeX를 유니코드로 변환
    md_content = convert_latex_to_unicode(md_content)
    
    # 이미지 경로를 절대 경로로 변환
    current_dir = os.path.abspath('.')
    md_content = md_content.replace('](images/', f']({current_dir}/images/')
    
    # Markdown을 HTML로 변환
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite'])
    html_content = md.convert(md_content)
    
    # CSS 스타일 정의 (한글 폰트 명시)
    css_style = """
    @font-face {
        font-family: 'Noto Sans CJK KR';
        src: local('Noto Sans CJK KR'), local('NotoSansCJK-Regular');
    }
    
    body {
        font-family: 'Noto Sans CJK KR', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
        max-width: 100%;
        margin: 0 auto;
        padding: 20px;
    }
    
    h1 {
        font-size: 24pt;
        color: #1a5f7a;
        border-bottom: 3px solid #1a5f7a;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    h2 {
        font-size: 18pt;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin-top: 25px;
    }
    
    h3 {
        font-size: 14pt;
        color: #34495e;
        margin-top: 20px;
    }
    
    h4 {
        font-size: 12pt;
        color: #555;
        margin-top: 15px;
    }
    
    p {
        text-align: justify;
        margin-bottom: 10px;
    }
    
    code {
        font-family: 'Consolas', 'Monaco', monospace;
        background-color: #f4f4f4;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 9pt;
    }
    
    pre {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        overflow-x: auto;
        font-size: 8pt;
        line-height: 1.4;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        font-size: 10pt;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    
    th {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 15px auto;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        padding-left: 15px;
        margin: 15px 0;
        color: #555;
        font-style: italic;
    }
    
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 20px 0;
    }
    
    ul, ol {
        margin-left: 20px;
        margin-bottom: 10px;
    }
    
    li {
        margin-bottom: 5px;
    }
    
    strong {
        color: #2c3e50;
    }
    
    /* 페이지 설정 */
    @page {
        size: A4;
        margin: 2cm;
        
        @bottom-center {
            content: counter(page);
            font-size: 10pt;
        }
    }
    
    /* 페이지 나눔 방지 */
    h1, h2, h3, h4 {
        page-break-after: avoid;
    }
    
    pre, table, img {
        page-break-inside: avoid;
    }
    """
    
    # 완전한 HTML 문서 생성
    full_html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>IMS 제2장: 연구 설계</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # HTML 파일 저장 (디버깅용)
    with open('Chapter_02_Study_Design.html', 'w', encoding='utf-8') as f:
        f.write(full_html)
    print("✓ HTML 파일 생성 완료")
    
    # PDF 생성
    html = HTML(string=full_html, base_url='.')
    css = CSS(string=css_style)
    html.write_pdf('Chapter_02_Study_Design.pdf', stylesheets=[css])
    
    print("✓ PDF 파일 생성 완료: Chapter_02_Study_Design.pdf")
    print("=" * 50)

if __name__ == '__main__':
    main()
