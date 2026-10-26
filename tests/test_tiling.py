import numpy as np
import pytest

from crackvision.data.tiling import (
    iou_matrix,
    merge_tile_detections,
    nms,
    pad_to_min,
    stitch_maps,
    tile_grid,
    tile_positions,
)


@pytest.mark.parametrize("length,tile", [(2016, 512), (1512, 512), (512, 512), (700, 512)])
def test_tile_positions_cover_the_whole_length(length, tile):
    positions = tile_positions(length, tile)
    assert positions[0] == 0
    assert positions[-1] + tile >= length
    covered = np.zeros(length, dtype=bool)
    for start in positions:
        covered[start:start + tile] = True
    assert covered.all()
    assert all(0 <= p <= max(length - tile, 0) for p in positions)


def test_tile_grid_counts():
    grid = tile_grid(1512, 2016, 512)
    assert len(grid) == 3 * 4
    assert {(r, c) for r, c, _, _ in grid} == {(r, c) for r in range(3) for c in range(4)}


def test_pad_to_min_keeps_original_size():
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    padded, (h, w) = pad_to_min(image, 192)
    assert padded.shape == (192, 300, 3)
    assert (h, w) == (100, 300)


def test_iou_and_nms():
    boxes = np.array([[0, 0, 10, 10, 0.9], [1, 1, 10, 10, 0.8], [20, 20, 30, 30, 0.7]],
                     dtype=np.float32)
    ious = iou_matrix(boxes[:1, :4], boxes[:, :4])
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 2] == 0
    kept = nms(boxes, 0.5)
    assert len(kept) == 2
    assert kept[0, 4] == pytest.approx(0.9)


def test_merge_tile_detections_unites_a_crack_split_by_two_tiles():
    # Meme fissure vue par deux tuiles : une boite contient presque l'autre.
    dets = np.array([[100, 100, 300, 140, 0.8], [250, 102, 298, 138, 0.6]], dtype=np.float32)
    merged = merge_tile_detections(dets)
    assert len(merged) == 1
    assert merged[0, :4].tolist() == [100, 100, 300, 140]


def test_stitch_maps_takes_the_maximum_on_overlaps():
    maps = [np.full((4, 4), 1.0, dtype=np.float32), np.full((4, 4), 2.0, dtype=np.float32)]
    canvas = stitch_maps(maps, [(0, 0), (0, 2)], (4, 6))
    assert canvas[0, 0] == 1.0
    assert canvas[0, 3] == 2.0
    assert canvas[0, 5] == 2.0
