from multiprocessing import Process, Queue
import numpy as np
import time

queue = Queue()

def gui_process(queue):
    import queue as queue_module

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # Each element is one complete frame:
    #
    # [
    #     ((x, y), color, shape),
    #     ((x, y), color, shape),
    #     ...
    # ]
    frames = []

    index = [0]

    # Global bounds across ALL frames received
    bounds = {
        "xmin": None,
        "xmax": None,
        "ymin": None,
        "ymax": None,
    }

    # ---------------------------------------------------------
    # GUI
    # ---------------------------------------------------------

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    prev_ax = fig.add_axes([0.25, 0.05, 0.2, 0.075])
    next_ax = fig.add_axes([0.55, 0.05, 0.2, 0.075])

    prev_button = Button(prev_ax, "Previous")
    next_button = Button(next_ax, "Next")

    # ---------------------------------------------------------
    # Bounds
    # ---------------------------------------------------------

    def update_bounds(points):
        if not points:
            return

        xs = [point[0][0] for point in points]
        ys = [point[0][1] for point in points]

        xmin = min(xs)
        xmax = max(xs)

        ymin = min(ys)
        ymax = max(ys)

        if bounds["xmin"] is None:
            bounds["xmin"] = xmin
            bounds["xmax"] = xmax
            bounds["ymin"] = ymin
            bounds["ymax"] = ymax

        else:
            bounds["xmin"] = min(bounds["xmin"], xmin)
            bounds["xmax"] = max(bounds["xmax"], xmax)
            bounds["ymin"] = min(bounds["ymin"], ymin)
            bounds["ymax"] = max(bounds["ymax"], ymax)

    # ---------------------------------------------------------
    # Render current frame
    # ---------------------------------------------------------

    def update():
        if not frames:
            return

        ax.clear()

        points = frames[index[0]]

        for (x, y), color, shape in points:

            # r = red
            # anything else = blue
            c = "red" if color == "r" else "blue"

            # True  = circle
            # False = X
            marker = "o" if shape else "x"

            ax.scatter(
                x,
                y,
                c=c,
                marker=marker,
                s=50,
            )

        # ---------------------------------------------
        # SAME scale for every frame
        # ---------------------------------------------

        xmin = bounds["xmin"]
        xmax = bounds["xmax"]
        ymin = bounds["ymin"]
        ymax = bounds["ymax"]

        if xmin is not None:

            # Add some padding
            x_range = xmax - xmin
            y_range = ymax - ymin

            x_padding = x_range * 0.1 if x_range else 1
            y_padding = y_range * 0.1 if y_range else 1

            ax.set_xlim(
                xmin - x_padding,
                xmax + x_padding,
            )

            ax.set_ylim(
                ymin - y_padding,
                ymax + y_padding,
            )

        ax.set_title(
            f"{index[0] + 1} / {len(frames)}"
        )

        ax.grid(True)

        fig.canvas.draw_idle()

    # ---------------------------------------------------------
    # Buttons
    # ---------------------------------------------------------

    def previous(event):
        if index[0] > 0:
            index[0] -= 1
            update()

    def next_frame(event):
        if index[0] < len(frames) - 1:
            index[0] += 1
            update()

    prev_button.on_clicked(previous)
    next_button.on_clicked(next_frame)

    # ---------------------------------------------------------
    # Read queue
    # ---------------------------------------------------------

    def check_queue():
        changed = False

        while True:
            try:
                new_frame = queue.get_nowait()
            except queue_module.Empty:
                break

            # Update GLOBAL bounds
            update_bounds(new_frame)

            frames.append(new_frame)

            changed = True

        # Important:
        #
        # Re-render even if we're looking at an OLD frame,
        # because the global bounds may have changed.
        if changed:
            update()

    # Check queue every 100ms
    timer = fig.canvas.new_timer(interval=100)

    timer.add_callback(check_queue)
    timer.start()

    plt.show()