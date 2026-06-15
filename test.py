# Run this to generate visualization you can show in video
import matplotlib.pyplot as plt
import numpy as np

# IoU bar chart
lesions = ['Ridge\n(Most Important)', 'Vessels\n(Moderate)', 'Optic Disc\n(Reference)']
iou_values = [0.183, 0.105, 0.067]
colors = ['#2ecc71', '#f39c12', '#e74c3c']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(lesions, iou_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Expected Range (0.05-0.30)')
ax.set_ylabel('IoU Score', fontsize=14, fontweight='bold')
ax.set_title('Quantitative XAI Validation: AI Attention vs Expert Annotations', 
             fontsize=16, fontweight='bold')
ax.set_ylim(0, 0.25)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('demo_iou_chart.png', dpi=300)
plt.show()

print("✓ Saved demo_iou_chart.png - Use this in your video!")