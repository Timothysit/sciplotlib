import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, TextColumn, BarColumn
from rich.progress import track
import typer
import ttkbootstrap as ttk
from ttkbootstrap import Style
# import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import sys
import pdb
import numpy as np
import os
import pickle

class ResizablePanel:
    def __init__(self, canvas, x, y, w, h, label="A"):
        self.canvas = canvas
        self.rect = canvas.create_rectangle(x, y, x + w, y + h, fill="lightgray")

        # Figure lettering
        self.label_id = canvas.create_text(x + 6, y + 6, text=label, anchor="nw", font=("Helvetica", 12, "bold"))

        self.drag_data = {"x": 0, "y": 0}
        self._bind_events()

        self.edge_margin = 6
        self.active_edge = None
        self.dragging = False

        # CONTEXT MENU (WHEN YOU RIGHT CLICK)
        self.menu = tk.Menu(canvas, tearoff=0)



        # ASSIGN FILE TO FIGURE PANEL
        self.menu.add_command(label="Assign file", command=self.assign_file)
        self.file_label_id = None

    def _bind_events(self):
        self.canvas.tag_bind(self.rect, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.rect, "<B1-Motion>", self.on_drag)

        if sys.platform == 'darwin':
            right_click_button_number = 2
        else:
            right_click_button_number = 3

        # Bind right-click  to open context menu
        self.canvas.tag_bind(self.rect, "<Button-%.f>" % right_click_button_number, self.show_context_menu)
        self.canvas.tag_bind(self.label_id, "<Button-%.f>" % right_click_button_number, self.show_context_menu)

        # Detect cursor change
        self.canvas.tag_bind(self.rect, "<Motion>", self.on_motion)

    def _detect_edge(self, x, y):
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        m = self.edge_margin

        near_left = abs(x - x0) < m
        near_right = abs(x - x1) < m
        near_top = abs(y - y0) < m
        near_bottom = abs(y - y1) < m

        # Corner priority
        if near_left and near_top:
            return "top-left"
        elif near_right and near_top:
            return "top-right"
        elif near_left and near_bottom:
            return "bottom-left"
        elif near_right and near_bottom:
            return "bottom-right"
        elif near_left:
            return "left"
        elif near_right:
            return "right"
        elif near_top:
            return "top"
        elif near_bottom:
            return "bottom"
        else:
            return None

    def _get_cursor(self, edge):
        return {
            "top": "sb_v_double_arrow",
            "bottom": "sb_v_double_arrow",
            "left": "sb_h_double_arrow",
            "right": "sb_h_double_arrow",
            "top-left": "top_left_corner",
            "top-right": "top_right_corner",
            "bottom-left": "bottom_left_corner",
            "bottom-right": "bottom_right_corner"
        }.get(edge, "arrow")

    def on_motion(self, event):
        edge = self._detect_edge(event.x, event.y)
        cursor = self._get_cursor(edge)
        self.canvas.config(cursor=cursor)

    def on_press(self, event):
        # Dragging the rectangle
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

        # Resizing the rectangle
        self.active_edge = self._detect_edge(event.x, event.y)
        if self.active_edge is None:
            self.dragging = True

    def on_drag(self, event):
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        x0, y0, x1, y1 = self.canvas.coords(self.rect)

        min_size = 30

        if self.active_edge == "left":
            x0 = min(x1 - min_size, x0 + dx)
        elif self.active_edge == "right":
            x1 = max(x0 + min_size, x1 + dx)
        elif self.active_edge == "top":
            y0 = min(y1 - min_size, y0 + dy)
        elif self.active_edge == "bottom":
            y1 = max(y0 + min_size, y1 + dy)
        elif self.active_edge == "top-left":
            x0 = min(x1 - min_size, x0 + dx)
            y0 = min(y1 - min_size, y0 + dy)
        elif self.active_edge == "top-right":
            x1 = max(x0 + min_size, x1 + dx)
            y0 = min(y1 - min_size, y0 + dy)
        elif self.active_edge == "bottom-left":
            x0 = min(x1 - min_size, x0 + dx)
            y1 = max(y0 + min_size, y1 + dy)
        elif self.active_edge == "bottom-right":
            x1 = max(x0 + min_size, x1 + dx)
            y1 = max(y0 + min_size, y1 + dy)
        elif self.dragging:
            # This is the dragging (moving the rectangle around) component
            self.canvas.move(self.rect, dx, dy)
            self.canvas.move(self.label_id, dx, dy)  # also move the text label along with it
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

            # Update file label position
            if self.file_label_id:
                x0, y0, x1, y1 = self.canvas.coords(self.rect)
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                self.canvas.coords(self.file_label_id, cx, cy)

            return

        self.canvas.coords(self.rect, x0, y0, x1, y1)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_release(self, event):
        self.active_edge = None
        self.dragging = False

    def on_resize_press(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_resize_drag(self, event):
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        self.canvas.coords(self.label_id, x0 + 6, y0 + 6)  # move the lettering as well with resize
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.canvas.coords(self.rect, x0, y0, x1 + dx, y1 + dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

        # Update file label position
        if self.file_label_id:
            x0, y0, x1, y1 = self.canvas.coords(self.rect)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            self.canvas.coords(self.file_label_id, cx, cy)

    def get_bbox(self):
        return self.canvas.coords(self.rect)  # [x0, y0, x1, y1]

    def show_context_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def assign_file(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.svg"),
                     ("Pickle files", "*.pkl"),
                     ("All files", "*.*")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.display_file_path(filepath)

    def display_file_path(self, filepath):
        # Save internally if needed
        self.filepath = filepath

        # Get current bounding box
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        if self.file_label_id:
            self.canvas.delete(self.file_label_id)

        self.file_label_id = self.canvas.create_text(
            cx, cy, text=filepath, anchor="center", font=("Helvetica", 9), width=x1 - x0 - 20
        )


def cm_to_px(cm, dpi=96):
    return int(cm * dpi / 2.54)

class FigureLayoutApp(ttk.Window):
    def __init__(self, PAPER_WIDTH_CM, PAPER_HEIGHT_CM, dpi):
        super().__init__(themename='lumen')
        self.title("Figure Layout GUI")
        self.geometry("1300x1300")

        ############################## Layout Frames ##############################
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        # Left side: Canvas
        self.canvas = tk.Canvas(main_frame, bg="white", width=800, height=1300)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Make sure grid expands canvas but not control panel
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        ### MAKE RIGHT PANEL ####
        # Right side: Vertical stack of control panels
        right_column_frame = ttk.Frame(main_frame)
        right_column_frame.grid(row=0, column=1, sticky="n", padx=(30, 50), pady=50)

        # Right side: Control Panel : Layout Controls
        control_panel = ttk.LabelFrame(right_column_frame, text="Layout controls", padding=10)
        control_panel.pack(fill="x", pady=(0, 30))
        # control_panel.grid(row=0, column=1, sticky="n", padx=100, pady=50)

        # Add buttons to control panel
        self.add_button = ttk.Button(control_panel, text="Add Panel", command=self.add_panel)
        self.add_button.pack(pady=(0, 10), fill="x")

        self.save_button = ttk.Button(control_panel, text="Save Layout", command=self.save_layout)
        self.save_button.pack(pady=(0, 10), fill="x")

        self.make_button = ttk.Button(control_panel, text="Make Figures", command=self.make_figures)
        self.make_button.pack(pady=(10, 0), fill="x")

        # Right side: Control Panel : Style controls
        style_panel = ttk.LabelFrame(right_column_frame, text="Style", padding=10)
        style_panel.pack(fill="x")
        # style_panel.grid(row=1, column=1, sticky="n", padx=100, pady=50)

        # 1. Stylesheet Dropdown
        ttk.Label(style_panel, text="Stylesheet").pack(anchor="w")
        self.stylesheet_var = tk.StringVar(value="None")
        ttk.Combobox(style_panel, textvariable=self.stylesheet_var, values=["None", "Modern", "nature-reviews"]).pack(
            fill="x", pady=(0, 10))

        # 2. Font Dropdown
        ttk.Label(style_panel, text="Font").pack(anchor="w")
        self.font_var = tk.StringVar(value="Helvetica")
        ttk.Combobox(style_panel, textvariable=self.font_var, values=["Helvetica", "Computer Modern Sans Serif"]).pack(
            fill="x", pady=(0, 10))

        # 3. Font Size (float entry)
        ttk.Label(style_panel, text="Font size").pack(anchor="w")
        self.font_size_var = tk.DoubleVar(value=11.0)
        ttk.Entry(style_panel, textvariable=self.font_size_var).pack(fill="x", pady=(0, 10))

        # 4. Tick mark font size
        ttk.Label(style_panel, text="Tick mark font size").pack(anchor="w")
        self.tick_font_size_var = tk.DoubleVar(value=9.0)
        ttk.Entry(style_panel, textvariable=self.tick_font_size_var).pack(fill="x")


        ########################### MAKE THE PAPER PAGE ######################################
        paper_width_px = cm_to_px(PAPER_WIDTH_CM, dpi)
        paper_height_px = cm_to_px(PAPER_HEIGHT_CM, dpi)

        # Offset to center the paper rectangle in the canvas
        offset_x = 50
        offset_y = 50
        self.paper_rect = self.canvas.create_rectangle(
            offset_x,
            offset_y,
            offset_x + paper_width_px,
            offset_y + paper_height_px,
            fill="white",
            outline="black",
            width=2
        )


        self.panels = []


    def add_panel(self):
        label = chr(ord('A') + len(self.panels))  # A, B, C, ...
        panel = ResizablePanel(self.canvas, 50, 50, 200, 150, label=label)
        self.panels.append(panel)


    def save_layout(self):

        return None

    def make_figures(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import pickle
        from pathlib import Path

        fig_width = self.canvas.bbox(self.paper_rect)[2] - self.canvas.bbox(self.paper_rect)[0]
        fig_height = self.canvas.bbox(self.paper_rect)[3] - self.canvas.bbox(self.paper_rect)[1]

        dpi = 100 # you can use self.dpi if you store it
        fig = plt.figure(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)

        # Get paper bounding box
        paper_x0, paper_y0, paper_x1, paper_y1 = self.canvas.coords(self.paper_rect)
        paper_w = paper_x1 - paper_x0
        paper_h = paper_y1 - paper_y0

        for panel in self.panels:
            x0, y0, x1, y1 = panel.get_bbox()
            rel_x = (x0 - paper_x0) / paper_w
            rel_y = (y0 - paper_y0) / paper_h
            rel_w = (x1 - x0) / paper_w
            rel_h = (y1 - y0) / paper_h

            # NOTE: subfig is actually an matplotlib axes object, may think about renaming it later to make it more clear
            # subfig = fig.add_axes([rel_x, 1 - rel_y - rel_h, rel_w, rel_h])  # note: matplotlib origin is bottom-left

            # subfigure may not work, due to requiring subplotspec (so may not support arbitrary placement) 
            # subfig = fig.add_subfigure([rel_x, 1 - rel_y - rel_h, rel_w, rel_h])  # note: matplotlib origin is bottom-left

            filepath = getattr(panel, "filepath", None)
            if filepath:
                suffix = Path(filepath).suffix.lower()
                if suffix in [".png", ".jpg", ".jpeg", ".svg"]:
                    img = mpimg.imread(filepath)
                    subfig.imshow(img)
                    subfig.axis("off")
                elif suffix == ".pkl":
                    try:
                        with open(filepath, "rb") as f:
                            sub_fig = pickle.load(f)
                        if isinstance(sub_fig, plt.Figure):
                            subfig.text(0.5, 0.5, "Embedded Figure", ha="center")

                            # Transer figure axes into our subfig object
                            src_axes = embedded_fig.get_axes()
                            if len(src_axes) == 1:
                                ax = subfig.add_subplot(1, 1, 1)
                                self.clone_axes_content(src_axes[0], ax)
                            else:
                                # You can recreate layout using subfig.subplots or subfig.add_subplot()
                                rows = 1
                                cols = len(src_axes)
                                axes_grid = subfig.subplots(rows, cols)
                                for src_ax, dest_ax in zip(src_axes, axes_grid):
                                    self.clone_axes_content(src_ax, dest_ax)

                    except Exception as e:
                        subfig.text(0.5, 0.5, f"Failed to load: {Path(filepath).name}", ha="center")
            else:
                subfig.set_xticks([])
                subfig.set_yticks([])
                subfig.set_facecolor("#f0f0f0")
                subfig.text(0.5, 0.5, "Empty", ha="center", va="center")

        fig.tight_layout()
        fig.suptitle("Composed Layout", fontsize=14)
        fig.show()



app = typer.Typer()


@app.command()
def make_layout(paper_size='a4', dpi: int = 96):

    if paper_size == 'a4':
        PAPER_WIDTH_CM = 21
        PAPER_HEIGHT_CM = 29.7

    # style = Style(theme='lumen')
    # ctk.set_appearance_mode("System")
    figureApp = FigureLayoutApp(PAPER_WIDTH_CM, PAPER_HEIGHT_CM, dpi)
    figureApp.mainloop()

@app.command()
def make_example_figures(save_folder=None):

    # Example figure 1 : scatter plot
    fig, ax = plt.subplots()
    x_vals = np.random.normal(0, 2, 100)
    y_vals = np.random.normal(0, 1, 100)

    ax.scatter(x_vals, y_vals)

    save_name = 'example_scatter.pkl'
    if save_folder is not None:
        save_path = os.path.join(save_folder, save_name)
    else:
        save_path = save_name

    with open(save_path, 'wb') as f:  # should be 'wb' rather than 'w'
        pickle.dump(fig, f)

    # Example figure 2 : imshow
    fig, ax = plt.subplots()
    with cbook.get_sample_data('grace_hopper.jpg') as image_file:
        image = plt.imread(image_file)

    ax.imshow(image)

    save_name = 'example_image.pkl'
    if save_folder is not None:
        save_path = os.path.join(save_folder, save_name)
    else:
        save_path = save_name

    with open(save_path, 'wb') as f:  # should be 'wb' rather than 'w'
        pickle.dump(fig, f)

    # Example figure 3: subplots of two things
    fig, axs = plt.subplots(1, 2)

    # bar plot
    fruits = ['apple', 'blueberry', 'cherry', 'orange']
    counts = [40, 100, 30, 55]
    bar_labels = ['red', 'blue', '_red', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    axs[0].bar(fruits, counts, label=bar_labels, color=bar_colors)

    # line plot
    x = np.linspace(0, 10, 1000)
    axs[1].plot(x, np.sin(x))

    save_name = 'example_subplots.pkl'
    if save_folder is not None:
        save_path = os.path.join(save_folder, save_name)
    else:
        save_path = save_name

    with open(save_path, 'wb') as f:  # should be 'wb' rather than 'w'
        pickle.dump(fig, f)


@app.command()
def main():

    logger.info('Running layout')

    # TODO: ask the user which process to run : make_layout


if __name__ == "__main__":
    app()