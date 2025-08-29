import os

IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'background.png'))

VERTICES = {
    "left_tactile_little_finger_tip": [(764, 166), (818, 166), (818, 219), (764, 219)],  # 54x53
    'left_tactile_little_finger_nail': [(730, 238), (836, 238), (836, 380), (730, 380)],  # 106x142
    'left_tactile_little_finger_pad': [(710, 468), (817, 468), (817, 593), (710, 593)],  # 107x125
    "left_tactile_ring_finger_tip": [(604, 79), (658, 79), (658, 132), (604, 132)],  # 54x53
    'left_tactile_ring_finger_nail': [(587, 148), (693, 148), (693, 290), (587, 290)],  # 106x142
    'left_tactile_ring_finger_pad': [(550, 468), (657, 468), (657, 593), (550, 593)],  # 107x125
    "left_tactile_middle_finger_tip": [(444, 43), (498, 43), (498, 96), (444, 96)],  # 54x53
    'left_tactile_middle_finger_nail': [(410, 132), (516, 132), (516, 274), (410, 274)],  # 106x142
    'left_tactile_middle_finger_pad': [(410, 468), (517, 468), (517, 593), (410, 593)],  # 107x125
    "left_tactile_index_finger_tip": [(267, 96), (321, 96), (321, 149), (267, 149)],  # 54x53
    'left_tactile_index_finger_nail': [(250, 168), (356, 168), (356, 310), (250, 310)],  # 106x142
    'left_tactile_index_finger_pad': [(267, 468), (374, 468), (374, 593), (267, 593)],  # 107x125
    "left_tactile_thumb_tip": [(109, 362), (163, 362), (163, 415), (109, 415)],  # 54x53
    'left_tactile_thumb_nail': [(73, 433), (179, 433), (179, 575), (73, 575)],  # 106x142
    "left_tactile_thumb_middle": [(109, 645), (163, 645), (163, 698), (109, 698)],  # 54x53
    "left_tactile_thumb_pad": [(75, 770), (216, 770), (216, 912), (75, 912)],  # 141x142
    'left_tactile_palm': [(285, 682), (747, 682), (747, 947), (285, 947)],  # 462x264
    "right_tactile_ring_finger_tip": [(1209, 79), (1263, 79), (1263, 132), (1209, 132)],  # 54x53
    'right_tactile_little_finger_nail': [(1033, 238), (1139, 238), (1139, 380), (1033, 380)],  # 106x142
    'right_tactile_little_finger_pad': [(1050, 468), (1157, 468), (1157, 593), (1050, 593)],  # 107x125
    "right_tactile_middle_finger_tip": [(1369, 43), (1423, 43), (1423, 96), (1369, 96)],  # 54x53
    'right_tactile_ring_finger_nail': [(1174, 148), (1280, 148), (1280, 290), (1174, 290)],  # 106x142
    'right_tactile_ring_finger_pad': [(1210, 468), (1317, 468), (1317, 593), (1210, 593)],  # 107x125
    "right_tactile_little_finger_tip": [(1049, 166), (1103, 166), (1103, 219), (1049, 219)],  # 54x53
    'right_tactile_middle_finger_nail': [(1351, 132), (1457, 132), (1457, 274), (1351, 274)],  # 106x142
    'right_tactile_middle_finger_pad': [(1350, 468), (1457, 468), (1457, 593), (1350, 593)],  # 107x125
    "right_tactile_index_finger_tip": [(1546, 96), (1600, 96), (1600, 149), (1546, 149)],  # 54x53
    'right_tactile_index_finger_nail': [(1511, 168), (1617, 168), (1617, 310), (1511, 310)],  # 106x142
    'right_tactile_index_finger_pad': [(1493, 468), (1600, 468), (1600, 593), (1493, 593)],  # 107x125
    "right_tactile_thumb_tip": [(1704, 362), (1758, 362), (1758, 415), (1704, 415)],  # 54x53
    'right_tactile_thumb_nail': [(1688, 433), (1794, 433), (1794, 575), (1688, 575)],  # 106x142
    "right_tactile_thumb_middle": [(1704, 645), (1758, 645), (1758, 698), (1704, 698)],  # 54x53
    "right_tactile_thumb_pad": [(1651, 770), (1792, 770), (1792, 912), (1651, 912)],  # 141x142
    'right_tactile_palm': [(1120, 682), (1582, 682), (1582, 946), (1118, 946)],  # 462x264
}


def split_vertice(vertices, image_shape):
    """
    vertices: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]  # axis-aligned rectangle assumed
    image_shape: (C, H, W)  # channel, height, width

    return: list of list of vertices (H x W)
    """
    _, H, W = image_shape

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_edges = [round(x_min + (x_max - x_min) * c / W) for c in range(W + 1)]
    y_edges = [round(y_min + (y_max - y_min) * r / H) for r in range(H + 1)]

    sub_rects = []
    for r in range(H):
        row = []
        for c in range(W):
            x0, x1 = x_edges[c], x_edges[c + 1]
            y0, y1 = y_edges[r], y_edges[r + 1]
            row.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        sub_rects.append(row)

    return sub_rects


if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    tactile_to_image_shape = {
        "left_tactile_little_finger_tip": (3, 3, 3),
        "left_tactile_little_finger_nail": (3, 12, 8),
        "left_tactile_little_finger_pad": (3, 10, 8),
        "left_tactile_ring_finger_tip": (3, 3, 3),
        "left_tactile_ring_finger_nail": (3, 12, 8),
        "left_tactile_ring_finger_pad": (3, 10, 8),
        "left_tactile_middle_finger_tip": (3, 3, 3),
        "left_tactile_middle_finger_nail": (3, 12, 8),
        "left_tactile_middle_finger_pad": (3, 10, 8),
        "left_tactile_index_finger_tip": (3, 3, 3),
        "left_tactile_index_finger_nail": (3, 12, 8),
        "left_tactile_index_finger_pad": (3, 10, 8),
        "left_tactile_thumb_tip": (3, 3, 3),
        "left_tactile_thumb_nail": (3, 12, 8),
        "left_tactile_thumb_middle": (3, 3, 3),
        "left_tactile_thumb_pad": (3, 12, 8),
        "left_tactile_palm": (3, 8, 14),
        "right_tactile_little_finger_tip": (3, 3, 3),
        "right_tactile_little_finger_nail": (3, 12, 8),
        "right_tactile_little_finger_pad": (3, 10, 8),
        "right_tactile_ring_finger_tip": (3, 3, 3),
        "right_tactile_ring_finger_nail": (3, 12, 8),
        "right_tactile_ring_finger_pad": (3, 10, 8),
        "right_tactile_middle_finger_tip": (3, 3, 3),
        "right_tactile_middle_finger_nail": (3, 12, 8),
        "right_tactile_middle_finger_pad": (3, 10, 8),
        "right_tactile_index_finger_tip": (3, 3, 3),
        "right_tactile_index_finger_nail": (3, 12, 8),
        "right_tactile_index_finger_pad": (3, 10, 8),
        "right_tactile_thumb_tip": (3, 3, 3),
        "right_tactile_thumb_nail": (3, 12, 8),
        "right_tactile_thumb_middle": (3, 3, 3),
        "right_tactile_thumb_pad": (3, 12, 8),
        "right_tactile_palm": (3, 8, 14),
    }

    # Load background image
    img = plt.imread(IMAGE_PATH)
    cmap = matplotlib.colormaps['viridis']

    # Generate dummy tactile data [0,1]
    get_data = lambda x: np.random.random()

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img)
    ax.axis('off')

    # Generate patches
    patches = {}
    for name, verts in VERTICES.items():
        C, H, W = tactile_to_image_shape[name]
        cell_quads = split_vertice(verts, (C, H, W))

        cell_polys = [[None for _ in range(W)] for _ in range(H)]
        for r in range(H):
            for c in range(W):
                poly = Polygon(cell_quads[r][c], closed=True,
                               facecolor=(0, 0, 0, 0),
                               edgecolor='none',
                               alpha=0.8)
                ax.add_patch(poly)
                cell_polys[r][c] = poly
        patches[name] = cell_polys

    # Draw tactile regions
    try:
        while True:
            for name in VERTICES.keys():
                _, H, W = tactile_to_image_shape[name]

                for r in range(H):
                    for c in range(W):
                        data = get_data(name)
                        color = cmap(data)
                        patches[name][r][c].set_facecolor(color)

            fig.canvas.draw_idle()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
