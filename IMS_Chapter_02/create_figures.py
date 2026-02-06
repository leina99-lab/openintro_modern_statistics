#!/usr/bin/env python3
"""
IMS Chapter 2 그림 생성 스크립트
한글 폰트: Noto Sans CJK KR
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np

# 한글 폰트 설정
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
import matplotlib.font_manager as fm
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

def create_pop_to_sample():
    """그림 2.1: 모집단에서 표본으로"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 모집단 원
    pop_circle = Circle((3, 3), 2.5, fill=False, edgecolor='#2E86AB', linewidth=3)
    ax.add_patch(pop_circle)
    ax.text(3, 5.8, '모집단\n(Population)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 모집단 내 점들
    np.random.seed(42)
    n_pop = 50
    theta = np.random.uniform(0, 2*np.pi, n_pop)
    r = np.sqrt(np.random.uniform(0, 1, n_pop)) * 2.3
    pop_x = 3 + r * np.cos(theta)
    pop_y = 3 + r * np.sin(theta)
    
    # 선택된 점들 (빨간색)
    selected = np.random.choice(n_pop, 10, replace=False)
    
    for i in range(n_pop):
        if i in selected:
            ax.scatter(pop_x[i], pop_y[i], c='#E63946', s=80, zorder=5, edgecolors='darkred')
        else:
            ax.scatter(pop_x[i], pop_y[i], c='#457B9D', s=50, zorder=4, alpha=0.6)
    
    # 표본 원
    sample_circle = Circle((9, 3), 1.5, fill=False, edgecolor='#E63946', linewidth=3)
    ax.add_patch(sample_circle)
    ax.text(9, 4.8, '표본\n(Sample)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 표본 내 점들
    sample_theta = np.linspace(0, 2*np.pi, 10, endpoint=False) + np.pi/10
    sample_r = 0.8
    for i, t in enumerate(sample_theta):
        ax.scatter(9 + sample_r * np.cos(t), 3 + sample_r * np.sin(t), 
                   c='#E63946', s=80, zorder=5, edgecolors='darkred')
    
    # 화살표
    ax.annotate('', xy=(7, 3), xytext=(5.8, 3),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=3))
    ax.text(6.4, 3.5, '무작위 선택', ha='center', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('images/fig_2_1_pop_to_sample.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_1_pop_to_sample.png 생성 완료")

def create_biased_sample():
    """그림 2.2: 편향된 표본"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 모집단 원
    pop_circle = Circle((3, 3), 2.5, fill=False, edgecolor='#2E86AB', linewidth=3)
    ax.add_patch(pop_circle)
    ax.text(3, 5.8, '모집단', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 모집단 내 점들 - 일부는 "건강 관련 전공" (진한 색)
    np.random.seed(123)
    n_pop = 50
    theta = np.random.uniform(0, 2*np.pi, n_pop)
    r = np.sqrt(np.random.uniform(0, 1, n_pop)) * 2.3
    pop_x = 3 + r * np.cos(theta)
    pop_y = 3 + r * np.sin(theta)
    
    # 상단 절반은 건강 관련 전공 (진한 색)
    health_related = pop_y > 3
    
    for i in range(n_pop):
        if health_related[i]:
            ax.scatter(pop_x[i], pop_y[i], c='#1D3557', s=60, zorder=4)
        else:
            ax.scatter(pop_x[i], pop_y[i], c='#A8DADC', s=60, zorder=4, alpha=0.5)
    
    # 표본 원 - 편향된 선택
    sample_circle = Circle((9, 3), 1.5, fill=False, edgecolor='#E63946', linewidth=3)
    ax.add_patch(sample_circle)
    ax.text(9, 4.8, '편향된 표본', ha='center', va='center', fontsize=12, fontweight='bold', color='#E63946')
    
    # 표본 내 점들 - 대부분 진한 색 (건강 관련)
    sample_theta = np.linspace(0, 2*np.pi, 10, endpoint=False) + np.pi/10
    sample_r = 0.8
    for i, t in enumerate(sample_theta):
        if i < 8:  # 8개는 건강 관련
            color = '#1D3557'
        else:
            color = '#A8DADC'
        ax.scatter(9 + sample_r * np.cos(t), 3 + sample_r * np.sin(t), 
                   c=color, s=80, zorder=5, edgecolors='black')
    
    # 화살표
    ax.annotate('', xy=(7, 3), xytext=(5.8, 3),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=3))
    ax.text(6.4, 3.5, '편향된 선택', ha='center', va='center', fontsize=11, color='#E63946')
    
    # 범례
    ax.scatter([], [], c='#1D3557', s=60, label='건강 관련 전공')
    ax.scatter([], [], c='#A8DADC', s=60, alpha=0.5, label='기타 전공')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/fig_2_2_biased_sample.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_2_biased_sample.png 생성 완료")

def create_nonresponse_bias():
    """그림 2.3: 비응답 편향"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 모집단 원
    pop_circle = Circle((3, 3), 2.5, fill=False, edgecolor='#2E86AB', linewidth=3)
    ax.add_patch(pop_circle)
    ax.text(3, 5.8, '모집단', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 모집단 내 점들 - 일부는 "응답 가능" (진한 색), 일부는 "비응답" (회색)
    np.random.seed(456)
    n_pop = 50
    theta = np.random.uniform(0, 2*np.pi, n_pop)
    r = np.sqrt(np.random.uniform(0, 1, n_pop)) * 2.3
    pop_x = 3 + r * np.cos(theta)
    pop_y = 3 + r * np.sin(theta)
    
    # 무작위로 30%만 응답
    respondents = np.random.choice([True, False], n_pop, p=[0.3, 0.7])
    
    for i in range(n_pop):
        if respondents[i]:
            ax.scatter(pop_x[i], pop_y[i], c='#457B9D', s=60, zorder=4)
        else:
            ax.scatter(pop_x[i], pop_y[i], c='#D3D3D3', s=60, zorder=4, alpha=0.5)
    
    # 표본 원
    sample_circle = Circle((9, 3), 1.5, fill=False, edgecolor='#457B9D', linewidth=3)
    ax.add_patch(sample_circle)
    ax.text(9, 4.8, '응답자만 포함된\n표본', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 표본 내 점들 - 응답자만
    n_sample = sum(respondents)
    sample_theta = np.linspace(0, 2*np.pi, min(n_sample, 10), endpoint=False) + np.pi/10
    sample_r = 0.8
    for t in sample_theta:
        ax.scatter(9 + sample_r * np.cos(t), 3 + sample_r * np.sin(t), 
                   c='#457B9D', s=80, zorder=5, edgecolors='black')
    
    # 화살표
    ax.annotate('', xy=(7, 3), xytext=(5.8, 3),
                arrowprops=dict(arrowstyle='->', color='#457B9D', lw=3))
    ax.text(6.4, 3.7, '응답자만', ha='center', va='center', fontsize=11)
    ax.text(6.4, 3.2, '도달 가능', ha='center', va='center', fontsize=11)
    
    # 범례
    ax.scatter([], [], c='#457B9D', s=60, label='응답자')
    ax.scatter([], [], c='#D3D3D3', s=60, alpha=0.5, label='비응답자')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/fig_2_3_nonresponse_bias.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_3_nonresponse_bias.png 생성 완료")

def create_simple_stratified():
    """그림 2.4: 단순무작위표집과 층화표집"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for ax in axes:
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 5)
        ax.axis('off')
    
    # 상단: 단순무작위표집
    ax = axes[0]
    ax.text(6, 4.7, '단순무작위표집 (Simple Random Sampling)', ha='center', fontsize=14, fontweight='bold')
    
    # 모집단 박스
    rect = Rectangle((0.5, 0.5), 11, 3.5, fill=False, edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(rect)
    
    np.random.seed(42)
    n = 60
    x = np.random.uniform(1, 11, n)
    y = np.random.uniform(1, 3.5, n)
    selected = np.random.choice(n, 18, replace=False)
    
    for i in range(n):
        if i in selected:
            ax.scatter(x[i], y[i], c='#E63946', s=80, zorder=5, edgecolors='darkred')
        else:
            ax.scatter(x[i], y[i], c='#457B9D', s=50, zorder=4, alpha=0.6)
    
    ax.text(6, 0.1, '빨간 점: 무작위로 선택된 18개 케이스', ha='center', fontsize=11)
    
    # 하단: 층화표집
    ax = axes[1]
    ax.text(6, 4.7, '층화표집 (Stratified Sampling)', ha='center', fontsize=14, fontweight='bold')
    
    # 6개 층
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    strata_x = [1, 3, 5, 7, 9, 11]
    
    for s, (sx, color) in enumerate(zip(strata_x, colors)):
        # 층 박스
        rect = Rectangle((sx - 0.8, 0.5), 1.6, 3.5, fill=True, facecolor=color, 
                          edgecolor='black', linewidth=1, alpha=0.3)
        ax.add_patch(rect)
        
        # 층 내 점들
        np.random.seed(s + 100)
        n_stratum = 10
        x = np.random.uniform(sx - 0.6, sx + 0.6, n_stratum)
        y = np.random.uniform(1, 3.5, n_stratum)
        selected = np.random.choice(n_stratum, 3, replace=False)
        
        for i in range(n_stratum):
            if i in selected:
                ax.scatter(x[i], y[i], c='#E63946', s=80, zorder=5, edgecolors='darkred')
            else:
                ax.scatter(x[i], y[i], c='#457B9D', s=50, zorder=4, alpha=0.6)
    
    ax.text(6, 0.1, '각 층에서 3개씩 무작위 선택 (총 18개)', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('images/fig_2_4_simple_stratified.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_4_simple_stratified.png 생성 완료")

def create_cluster_multistage():
    """그림 2.5: 군집표집과 다단계표집"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for ax in axes:
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 5)
        ax.axis('off')
    
    # 상단: 군집표집
    ax = axes[0]
    ax.text(6, 4.7, '군집표집 (Cluster Sampling)', ha='center', fontsize=14, fontweight='bold')
    
    # 9개 군집
    cluster_positions = [(1.5, 3), (4, 3), (6.5, 3), (9, 3), (10.5, 3),
                        (1.5, 1.5), (4, 1.5), (6.5, 1.5), (9, 1.5)]
    selected_clusters = [1, 4, 7]  # 선택된 군집
    
    for c, (cx, cy) in enumerate(cluster_positions):
        circle = Circle((cx, cy), 0.8, fill=True, 
                        facecolor='#FFE4B5' if c in selected_clusters else '#E8E8E8',
                        edgecolor='#E63946' if c in selected_clusters else '#666666',
                        linewidth=2 if c in selected_clusters else 1)
        ax.add_patch(circle)
        
        # 군집 내 점들
        np.random.seed(c + 200)
        n_cluster = 6
        theta = np.linspace(0, 2*np.pi, n_cluster, endpoint=False)
        r = 0.4
        for t in theta:
            color = '#E63946' if c in selected_clusters else '#457B9D'
            alpha = 1.0 if c in selected_clusters else 0.4
            ax.scatter(cx + r * np.cos(t), cy + r * np.sin(t), 
                      c=color, s=50, zorder=5, alpha=alpha)
    
    ax.text(6, 0.3, '3개 군집 선택 → 선택된 군집의 모든 관측치 포함', ha='center', fontsize=11)
    
    # 하단: 다단계표집
    ax = axes[1]
    ax.text(6, 4.7, '다단계표집 (Multistage Sampling)', ha='center', fontsize=14, fontweight='bold')
    
    for c, (cx, cy) in enumerate(cluster_positions):
        circle = Circle((cx, cy), 0.8, fill=True, 
                        facecolor='#FFE4B5' if c in selected_clusters else '#E8E8E8',
                        edgecolor='#E63946' if c in selected_clusters else '#666666',
                        linewidth=2 if c in selected_clusters else 1)
        ax.add_patch(circle)
        
        # 군집 내 점들
        np.random.seed(c + 200)
        n_cluster = 6
        theta = np.linspace(0, 2*np.pi, n_cluster, endpoint=False)
        r = 0.4
        
        if c in selected_clusters:
            # 선택된 군집에서 일부만 선택
            selected_in_cluster = np.random.choice(n_cluster, 3, replace=False)
            for i, t in enumerate(theta):
                if i in selected_in_cluster:
                    ax.scatter(cx + r * np.cos(t), cy + r * np.sin(t), 
                              c='#E63946', s=60, zorder=5)
                else:
                    ax.scatter(cx + r * np.cos(t), cy + r * np.sin(t), 
                              c='#457B9D', s=40, zorder=5, alpha=0.5)
        else:
            for t in theta:
                ax.scatter(cx + r * np.cos(t), cy + r * np.sin(t), 
                          c='#457B9D', s=40, zorder=5, alpha=0.3)
    
    ax.text(6, 0.3, '3개 군집 선택 → 각 군집에서 일부만 무작위 선택', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('images/fig_2_5_cluster_multistage.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_5_cluster_multistage.png 생성 완료")

def create_blocking():
    """그림 2.6: 블록화"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    ax.text(6, 6.7, '블록화 (Blocking by Patient Risk)', ha='center', fontsize=14, fontweight='bold')
    
    # 저위험군 블록
    rect1 = FancyBboxPatch((0.5, 3.5), 5, 2.5, boxstyle="round,pad=0.05",
                           facecolor='#90EE90', edgecolor='#228B22', linewidth=2, alpha=0.5)
    ax.add_patch(rect1)
    ax.text(3, 5.7, '저위험군 블록', ha='center', fontsize=12, fontweight='bold', color='#228B22')
    
    # 고위험군 블록
    rect2 = FancyBboxPatch((0.5, 0.5), 5, 2.5, boxstyle="round,pad=0.05",
                           facecolor='#FFB6C1', edgecolor='#DC143C', linewidth=2, alpha=0.5)
    ax.add_patch(rect2)
    ax.text(3, 2.7, '고위험군 블록', ha='center', fontsize=12, fontweight='bold', color='#DC143C')
    
    # 환자 점들 - 저위험군
    np.random.seed(777)
    for i in range(8):
        x = 1 + (i % 4) * 1.1
        y = 4 + (i // 4) * 0.8
        ax.scatter(x, y, c='#228B22', s=100, zorder=5, marker='o')
    
    # 환자 점들 - 고위험군
    for i in range(8):
        x = 1 + (i % 4) * 1.1
        y = 1 + (i // 4) * 0.8
        ax.scatter(x, y, c='#DC143C', s=100, zorder=5, marker='o')
    
    # 화살표
    ax.annotate('', xy=(6.5, 4.75), xytext=(5.7, 4.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(6.5, 1.75), xytext=(5.7, 1.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.text(6.1, 3.25, '무작위\n배정', ha='center', va='center', fontsize=11)
    
    # 처치군 박스
    rect3 = FancyBboxPatch((7, 3.5), 2.2, 2.5, boxstyle="round,pad=0.05",
                           facecolor='#87CEEB', edgecolor='#4169E1', linewidth=2, alpha=0.5)
    ax.add_patch(rect3)
    ax.text(8.1, 5.7, '처치군', ha='center', fontsize=12, fontweight='bold', color='#4169E1')
    
    # 대조군 박스
    rect4 = FancyBboxPatch((9.5, 3.5), 2.2, 2.5, boxstyle="round,pad=0.05",
                           facecolor='#DDA0DD', edgecolor='#8B008B', linewidth=2, alpha=0.5)
    ax.add_patch(rect4)
    ax.text(10.6, 5.7, '대조군', ha='center', fontsize=12, fontweight='bold', color='#8B008B')
    
    # 처치군 - 저위험 2명, 고위험 2명
    for i, (x, y, c) in enumerate([(7.5, 4.8, '#228B22'), (8.3, 4.8, '#228B22'),
                                    (7.5, 4.0, '#DC143C'), (8.3, 4.0, '#DC143C')]):
        ax.scatter(x, y, c=c, s=100, zorder=5)
    
    # 대조군 - 저위험 2명, 고위험 2명
    for i, (x, y, c) in enumerate([(10, 4.8, '#228B22'), (10.8, 4.8, '#228B22'),
                                    (10, 4.0, '#DC143C'), (10.8, 4.0, '#DC143C')]):
        ax.scatter(x, y, c=c, s=100, zorder=5)
    
    # 범례
    ax.scatter([], [], c='#228B22', s=100, label='저위험 환자')
    ax.scatter([], [], c='#DC143C', s=100, label='고위험 환자')
    ax.legend(loc='lower center', fontsize=10, ncol=2)
    
    plt.tight_layout()
    plt.savefig('images/fig_2_6_blocking.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_6_blocking.png 생성 완료")

def create_confounding():
    """그림 2.7: 교란 변수"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(5, 5.7, '교란 변수 (Confounding Variable)', ha='center', fontsize=14, fontweight='bold')
    
    # 세 개의 박스
    def draw_box(x, y, text, color):
        box = FancyBboxPatch((x - 1.3, y - 0.5), 2.6, 1, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    draw_box(5, 4.5, '자외선 노출\n(교란 변수)', '#FFD700')
    draw_box(2, 1.5, '자외선 차단제\n사용', '#87CEEB')
    draw_box(8, 1.5, '피부암', '#FF6B6B')
    
    # 화살표
    # 자외선 노출 → 자외선 차단제
    ax.annotate('', xy=(2.5, 2.2), xytext=(4.3, 3.8),
                arrowprops=dict(arrowstyle='->', color='#228B22', lw=2.5))
    
    # 자외선 노출 → 피부암
    ax.annotate('', xy=(7.5, 2.2), xytext=(5.7, 3.8),
                arrowprops=dict(arrowstyle='->', color='#228B22', lw=2.5))
    
    # 자외선 차단제 → 피부암 (점선, 물음표)
    ax.annotate('', xy=(6.5, 1.5), xytext=(3.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#FF4500', lw=2, linestyle='dashed'))
    ax.text(5, 1.9, '인과관계?', ha='center', va='center', fontsize=11, color='#FF4500', fontweight='bold')
    
    ax.text(3, 3.3, '영향', ha='center', fontsize=10, color='#228B22')
    ax.text(7, 3.3, '영향', ha='center', fontsize=10, color='#228B22')
    
    plt.tight_layout()
    plt.savefig('images/fig_2_7_confounding.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_7_confounding.png 생성 완료")

def create_scope_of_inference():
    """그림 2.8: 추론의 범위"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.7, '추론의 범위 (Scope of Inference)', ha='center', fontsize=16, fontweight='bold')
    
    # 헤더
    ax.text(6, 6.8, '처치의 무작위 배정?', ha='center', fontsize=13, fontweight='bold')
    ax.text(3.5, 6.3, '예', ha='center', fontsize=12, fontweight='bold', color='#228B22')
    ax.text(8.5, 6.3, '아니오', ha='center', fontsize=12, fontweight='bold', color='#DC143C')
    
    # 행 헤더
    ax.text(0.3, 4.5, '무작위\n표본?', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    ax.text(1.2, 5.2, '예', ha='center', fontsize=12, fontweight='bold', color='#228B22')
    ax.text(1.2, 2.8, '아니오', ha='center', fontsize=12, fontweight='bold', color='#DC143C')
    
    # 4개 셀
    cells = [
        (3.5, 5.2, '인과관계 O\n일반화 O', '#90EE90', '(매우 드묾)'),
        (8.5, 5.2, '인과관계 X\n일반화 O', '#FFD700', ''),
        (3.5, 2.8, '인과관계 O\n일반화 제한적', '#87CEEB', ''),
        (8.5, 2.8, '인과관계 X\n일반화 제한적', '#FFB6C1', ''),
    ]
    
    for x, y, text, color, note in cells:
        rect = FancyBboxPatch((x - 2, y - 1), 4, 2, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.2, text, ha='center', va='center', fontsize=11, fontweight='bold')
        if note:
            ax.text(x, y - 0.6, note, ha='center', va='center', fontsize=10, fontstyle='italic', color='#666666')
    
    # 그리드 선
    ax.plot([1.5, 10.5], [4, 4], 'k-', lw=1)
    ax.plot([6, 6], [1.5, 6], 'k-', lw=1)
    
    # 설명
    ax.text(6, 0.8, '무작위 표본 → 모집단으로 일반화 가능\n무작위 배정 → 인과관계 확립 가능', 
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('images/fig_2_8_scope_of_inference.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig_2_8_scope_of_inference.png 생성 완료")

if __name__ == '__main__':
    print("=" * 50)
    print("IMS Chapter 2 그림 생성 시작")
    print("=" * 50)
    create_pop_to_sample()
    create_biased_sample()
    create_nonresponse_bias()
    create_simple_stratified()
    create_cluster_multistage()
    create_blocking()
    create_confounding()
    create_scope_of_inference()
    print("=" * 50)
    print("모든 그림 생성 완료!")
    print("=" * 50)
