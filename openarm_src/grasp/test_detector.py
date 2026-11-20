"""merge_duplicate_boxes 的單元測試（不需 GPU / OWLv2）。

執行：PYTHONPATH=$HOME/lerobot pytest openarm_src/grasp/test_detector.py
"""
from openarm_src.grasp.detector import Box, merge_duplicate_boxes


def test_overlapping_duplicates_keep_highest_score():
    out = merge_duplicate_boxes([Box(100, 20, 160, 230, 0.30), Box(102, 25, 162, 232, 0.45)])
    assert len(out) == 1 and out[0].score == 0.45


def test_partial_box_inside_full_bottle_keeps_full_box():
    """只框到標籤的小框分數較高：保留完整的瓶子框（分數取兩者較高者）。"""
    full = Box(100, 10, 160, 239, 0.30)
    label = Box(105, 100, 155, 160, 0.40)
    out = merge_duplicate_boxes([full, label])
    assert len(out) == 1
    assert (out[0].x0, out[0].y0, out[0].x1, out[0].y1) == (100, 10, 160, 239)
    assert out[0].score == 0.40


def test_low_score_big_box_does_not_swallow_bottle():
    """大框分數不到小框一半：多半是誤框，留小框。"""
    big = Box(50, 0, 250, 239, 0.12)
    bottle = Box(100, 20, 160, 230, 0.40)
    out = merge_duplicate_boxes([big, bottle])
    assert len(out) == 1 and (out[0].x0, out[0].x1) == (100, 160)


def test_specific_label_survives_merge():
    """同一支瓶子有一部分被認成指定的瓶子（label 0）：合併後的框也是 label 0。"""
    full = Box(100, 10, 160, 239, 0.35, label=1)
    logo = Box(110, 90, 150, 150, 0.30, label=0)
    out = merge_duplicate_boxes([full, logo])
    assert len(out) == 1 and out[0].label == 0 and out[0].x0 == 100


def test_two_side_by_side_bottles_not_merged_by_group_box():
    """框到並排兩支瓶子的大框被丟掉，兩支瓶子各自保留。"""
    a = Box(60, 20, 120, 230, 0.35)
    b = Box(200, 20, 260, 230, 0.30)
    group = Box(55, 15, 265, 235, 0.40)
    out = merge_duplicate_boxes([a, b, group])
    assert sorted((o.x0, o.x1) for o in out) == [(60, 120), (200, 260)]


def test_cap_and_label_of_same_bottle_do_not_drop_full_box():
    """瓶蓋框、標籤框上下排、互不重疊：仍是同一支瓶子，保留完整框。"""
    full = Box(100, 10, 160, 239, 0.30)
    cap = Box(115, 10, 145, 50, 0.20)
    label = Box(102, 110, 158, 170, 0.28)
    out = merge_duplicate_boxes([full, cap, label])
    assert len(out) == 1 and (out[0].y0, out[0].y1) == (10, 239)


def test_separate_bottles_untouched():
    a = Box(10, 20, 70, 230, 0.3)
    b = Box(300, 20, 360, 230, 0.2)
    assert len(merge_duplicate_boxes([a, b])) == 2
