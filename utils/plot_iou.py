import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def calculate_and_plot_iou(gt_box, pred_box):
  """
  Calculates IoU and plots the bounding boxes

  Args:
      gt_box: [x1, y1, x2, y2] - Ground Truth
      pred_box: [x1, y1, x2, y2] - Prediction
  """
  # Calculate intersection
  x1_inter = max(gt_box[0], pred_box[0])
  y1_inter = max(gt_box[1], pred_box[1])
  x2_inter = min(gt_box[2], pred_box[2])
  y2_inter = min(gt_box[3], pred_box[3])

  # Intersection area
  if x2_inter < x1_inter or y2_inter < y1_inter:
    intersection = 0
    inter_box = None
  else:
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    inter_box = [x1_inter, y1_inter, x2_inter, y2_inter]

  # Box areas
  area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
  area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

  # Union
  union = area_gt + area_pred - intersection

  # IoU
  iou = intersection / union if union > 0 else 0

  # Create figure
  fig, ax = plt.subplots(1, 1, figsize=(10, 10))

  # Ground Truth (green)
  gt_rect = patches.Rectangle(
      (gt_box[0], gt_box[1]),
      gt_box[2] - gt_box[0],
      gt_box[3] - gt_box[1],
      linewidth=3,
      edgecolor='green',
      facecolor='green',
      alpha=0.25,
      label='Ground Truth'
  )
  ax.add_patch(gt_rect)

  # Prediction (red)
  pred_rect = patches.Rectangle(
      (pred_box[0], pred_box[1]),
      pred_box[2] - pred_box[0],
      pred_box[3] - pred_box[1],
      linewidth=3,
      edgecolor='red',
      facecolor='red',
      alpha=0.25,
      label='Prediction'
  )
  ax.add_patch(pred_rect)

  # Intersection (blue)
  if inter_box is not None:
    inter_rect = patches.Rectangle(
        (inter_box[0], inter_box[1]),
        inter_box[2] - inter_box[0],
        inter_box[3] - inter_box[1],
        linewidth=2,
        edgecolor='blue',
        facecolor='blue',
        alpha=0.4,
        label='Intersection',
        linestyle='--'
    )
    ax.add_patch(inter_rect)

  # Fixed 224x224 grid
  ax.set_xlim(0, 224)
  ax.set_ylim(0, 224)
  ax.set_aspect('equal')
  ax.invert_yaxis()
  ax.grid(True, alpha=0.3, linestyle='--')
  ax.legend(loc='upper right', fontsize=12)

  status = "PASS" if iou >= 0.5 else "FAIL"

  plt.tight_layout()

  # Print to console
  print("="*80)
  print(f"IoU CALCULATION")
  print("="*80)
  print(f"Ground Truth:  {gt_box}")
  print(f"Prediction:    {pred_box}")
  print(f"\nGT Area:       {area_gt:.0f} px²")
  print(f"Pred Area:     {area_pred:.0f} px²")
  print(f"Intersection:  {intersection:.0f} px²")
  print(f"Union:         {union:.0f} px²")
  print(f"\nIoU = {intersection:.0f} / {union:.0f} = {iou:.4f} ({iou*100:.2f}%)")
  print(f"Status: {status}")
  print("="*80)

  return fig, iou

gt = [20, 20, 120, 120]    # 100x100
pred = [30, 30, 130, 130]  # Only shifted 10px (10% of size)

"""
# Example 1: ONLY 10% DISPLACEMENT - looks 90% correct, but IoU ~0.68
gt = [20, 20, 120, 120]    # 100x100
pred = [30, 30, 130, 130]  # Only shifted 10px (10% of size)
# Visually: "Wow, almost perfect!"
# Reality: IoU = 0.68 (if threshold is 0.7, FAILS!)

# Example 2: BOX 15% LARGER - looks ok, but IoU ~0.64
gt = [40, 40, 140, 140]    # 100x100
pred = [25, 25, 155, 155]  # 15px larger on each side
# Visually: "Caught everything and a bit more"
# Reality: IoU = 0.64

# Example 3: BOX 20% SMALLER - looks reasonable, but IoU ~0.64
gt = [20, 20, 180, 180]    # 160x160
pred = [52, 52, 148, 148]  # 20% smaller (96x96)
# Visually: "Got the center perfectly"
# Reality: IoU = 0.64

# Example 4: SMALL DIAGONAL DISPLACEMENT - looks 85% correct, but IoU ~0.62
gt = [30, 30, 150, 150]    # 120x120
pred = [45, 45, 165, 165]  # Diagonal 15px
# Visually: "Almost there!"
# Reality: IoU = 0.62

# Example 5: PERFECT WIDTH, 80% HEIGHT - looks good, but IoU ~0.64
gt = [50, 30, 170, 150]    # 120x120
pred = [50, 42, 170, 138]  # Height reduced 20%
# Visually: "Perfect width, almost right height!"
# Reality: IoU = 0.64

# Example 6: ONE SIDE PERFECT, OTHER DISPLACED - looks 80%, but IoU ~0.56
gt = [40, 40, 160, 160]    # 120x120
pred = [40, 55, 160, 175]  # Left side perfect, right displaced
# Visually: "3 of 4 sides correct!"
# Reality: IoU = 0.56

# Example 7: SMALL AREA - 8px displacement seems nothing, but IoU ~0.49
gt = [50, 50, 90, 90]      # 40x40
pred = [58, 58, 98, 98]    # Only 8px displacement (20% of size)
# Visually: "Practically on top!"
# Reality: IoU = 0.49 FAILS!

# Example 8: BOX 12% LARGER - looks like "caught everything", but IoU ~0.70
gt = [60, 60, 160, 160]    # 100x100
pred = [48, 48, 172, 172]  # 12px larger each side
# Visually: "Covered the whole object"
# Reality: IoU = 0.70

# Example 9: LARGE OBJECT - 20px displacement seems small, but IoU ~0.68
gt = [10, 10, 210, 210]    # 200x200
pred = [30, 30, 224, 224]  # Only 20px displacement (10% of size)
# Visually: "Almost perfect on large object"
# Reality: IoU = 0.68

# Example 10: RECTANGLE - width ok, 75% height, looks good but IoU ~0.60
gt = [30, 40, 150, 180]    # 120x140
pred = [30, 58, 150, 162]  # Perfect width, 75% height
# Visually: "Correct width, reasonable height"
# Reality: IoU = 0.60

# Example 11: 5px DISPLACEMENT ON SMALL OBJECT - seems nothing, but IoU ~0.56
gt = [80, 80, 120, 120]    # 40x40
pred = [85, 85, 125, 125]  # Only 5px! (12.5% of size)
# Visually: "Minimal displacement!"
# Reality: IoU = 0.56

# Example 12: BOX 8% LARGER - almost imperceptible, but IoU ~0.73
gt = [50, 50, 170, 170]    # 120x120
pred = [42, 42, 178, 178]  # 8px larger each side (7% of size)
# Visually: "Practically identical!"
# Reality: IoU = 0.73

# Example 13: TYPICAL WSOD - heatmap catches 85% but extrapolates, IoU ~0.58
gt = [40, 60, 140, 160]    # 100x100
pred = [30, 70, 150, 150]  # Width +20px, height -10px
# Visually: "Caught almost all the fire"
# Reality: IoU = 0.58

# Example 14: ASYMMETRIC DISPLACEMENT - looks 75%, but IoU ~0.52
gt = [50, 50, 150, 150]    # 100x100
pred = [60, 45, 160, 155]  # +10px X, -5px Y, slightly larger
# Visually: "Very close!"
# Reality: IoU = 0.52

# Example 15: MEDIUM OBJECT - box 18% smaller, looks ok but IoU ~0.67
gt = [30, 30, 190, 190]    # 160x160
pred = [59, 59, 161, 161]  # ~18% smaller each side
# Visually: "Got the core of the object"
# Reality: IoU = 0.67

# Example 16: ONE DIMENSION PERFECT - looks 90%, but IoU ~0.62
gt = [40, 50, 140, 170]    # 100x120
pred = [40, 65, 140, 155]  # X perfect, Y displaced
# Visually: "100% correct width!"
# Reality: IoU = 0.62

# Example 17: CONNECTED HEATMAP BOX - irregular shape, IoU ~0.48
gt = [60, 60, 140, 140]    # 80x80
pred = [55, 72, 145, 132]  # Slightly displaced and different shape
# Visually: "Covered the fire region"
# Reality: IoU = 0.48 FAILS!

# Example 18: LARGE - 15px displacement seems minimal, but IoU ~0.75
gt = [12, 12, 212, 212]    # 200x200
pred = [27, 27, 224, 224]  # Only 15px displacement (7.5%)
# Visually: "Insignificant displacement!"
# Reality: IoU = 0.75 (passes, but far from ideal)

# Example 19: THIN RECTANGLE - 6px displacement seems nothing, IoU ~0.54
gt = [80, 30, 120, 190]    # 40x160
pred = [86, 36, 126, 184]  # Only 6px displacement
# Visually: "Practically on top!"
# Reality: IoU = 0.54

# Example 20: BOX 10% LARGER ON SMALL OBJECT - IoU ~0.67
gt = [100, 100, 150, 150]  # 50x50
pred = [95, 95, 155, 155]  # 5px larger each side (10%)
# Visually: "Caught everything perfectly"
# Reality: IoU = 0.67
"""
fig1, iou1 = calculate_and_plot_iou(gt, pred)
plt.show()

#[2026-01-26 12:39:37] INFO:       GT Box (x1,y1,x2,y2): (88, 195, 233, 253)
# [2026-01-26 12:39:37] INFO:       Pred Box (x1,y1,x2,y2): (np.int32(92),
# np.int32(196), np.int32(256), np.int32(256))
