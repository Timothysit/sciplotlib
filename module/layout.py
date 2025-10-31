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
import io
import pickle
from pathlib import Path

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.image import AxesImage
from matplotlib.text import Text

# Sciplotlib
import sciplotlib.style as splstyle

# For saving
import json

import pdb
from copy import copy

class ResizablePanel:
    def __init__(self, canvas, x, y, w, h, label="A",
                 grid_rows=6, grid_cols=4, paper_bbox=None):
        self.canvas = canvas
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.paper_bbox = paper_bbox  # [x0, y0, x1, y1]
        self.rect = canvas.create_rectangle(x, y, x + w, y + h, fill="lightgray")

        # Figure lettering
        self.label_id = canvas.create_text(x + 6, y + 6, text=label, anchor="nw", font=("Helvetica", 12, "bold"))
        self.label_text = label
        # self.rect, self.label_id = self._draw_panel()

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

    def _draw_panel(self):
        rect = self.canvas.create_rectangle(self.x0, self.y0, self.x1, self.y1, fill="lightgray")
        label_id = self.canvas.create_text(self.x0 + 6, self.y0 + 6, text=self.label_text,
                                           anchor="nw", font=("Helvetica", 12, "bold"))
        return rect, label_id

    def snap_to_grid(self, x, y):
        if not self.paper_bbox:
            return x, y  # fallback: no snapping

        x0, y0, x1, y1 = self.paper_bbox
        w = x1 - x0
        h = y1 - y0

        cell_w = w / self.grid_cols
        cell_h = h / self.grid_rows

        col = round((x - x0) / cell_w)
        row = round((y - y0) / cell_h)

        return x0 + col * cell_w, y0 + row * cell_h

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

        # Bind button release
        self.canvas.tag_bind(self.rect, "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind(self.label_id, "<ButtonRelease-1>", self.on_release)

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


        min_size = 30
        x0, y0, x1, y1 = self.canvas.coords(self.rect)

        if self.active_edge:

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

            self.canvas.coords(self.rect, x0, y0, x1, y1)
            self.canvas.coords(self.label_id, x0 + 6, y0 + 6)
            if self.file_label_id:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                self.canvas.coords(self.file_label_id, cx, cy)

        elif self.dragging:
            self.canvas.move(self.rect, dx, dy)
            self.canvas.move(self.label_id, dx, dy)
            if self.file_label_id:
                self.canvas.move(self.file_label_id, dx, dy)

        # Original arbitrary coordinates
        # self.canvas.coords(self.rect, x0, y0, x1, y1)


        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_release(self, event):
        x0, y0, x1, y1 = self.canvas.coords(self.rect)

        # Snap both corners
        x0, y0 = self.snap_to_grid(x0, y0)
        x1, y1 = self.snap_to_grid(x1, y1)

        self.canvas.coords(self.rect, x0, y0, x1, y1)
        self.canvas.coords(self.label_id, x0 + 6, y0 + 6)
        if self.file_label_id:
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            self.canvas.coords(self.file_label_id, cx, cy)

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
        # self.canvas.coords(self.rect, x0, y0, x1 + dx, y1 + dy)

        # Snap corners to grid
        x0, y0 = self.snap_to_grid(x0, y0)
        x1, y1 = self.snap_to_grid(x1, y1)

        # Update rectangle
        self.canvas.coords(self.rect, x0, y0, x1, y1)

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

        # Quite app when window is closed
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.dpi = dpi 

        # Initial figure size 
        self.paper_width_cm = 0
        self.paper_height_cm = 0


        # Grid layout
        self.grid_rows_var = tk.IntVar(value=20)
        self.grid_cols_var = tk.IntVar(value=10)

        # Internal copies to update logic
        self.grid_rows = self.grid_rows_var.get()
        self.grid_cols = self.grid_cols_var.get()

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

        self.load_button = ttk.Button(control_panel, text="Load Layout", command=self.load_layout)
        self.load_button.pack(pady=(0, 10), fill="x")

        self.make_button = ttk.Button(control_panel, text="Make Figures", command=self.make_figures)
        self.make_button.pack(pady=(10, 0), fill="x")

        # --- SAVE FIGURE CONTROLS ---
        save_frame = ttk.LabelFrame(control_panel, text="Save Options", padding=5)
        save_frame.pack(fill='x', pady=(15, 0))

        ttk.Label(save_frame, text="Save Path").pack(anchor="w")

        path_entry_frame = ttk.Frame(save_frame)
        path_entry_frame.pack(fill='x', expand=True)

        self.save_path_var = tk.StringVar()
        # Set the default path to ~/composed_layout
        default_save_path = Path.home() / "composed_layout"
        self.save_path_var.set(str(default_save_path))

        save_entry = ttk.Entry(path_entry_frame, textvariable=self.save_path_var)
        save_entry.pack(side='left', fill='x', expand=True)

        browse_button = ttk.Button(path_entry_frame, text="...", command=self._browse_save_path, width=3)
        browse_button.pack(side='right', padx=(5, 0))

        # Grid customisation
        ttk.Label(control_panel, text="Grid Rows").pack(anchor="w")
        ttk.Entry(control_panel, textvariable=self.grid_rows_var).pack(fill="x", pady=(0, 10))

        ttk.Label(control_panel, text="Grid Columns").pack(anchor="w")
        ttk.Entry(control_panel, textvariable=self.grid_cols_var).pack(fill="x", pady=(0, 10))

        ttk.Button(control_panel, text="Update Grid", command=self.update_grid).pack(pady=(10, 0), fill="x")

        # Paper Size Controls
        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(control_panel, text="Paper Size").pack(anchor="w")
        self.paper_size_var = tk.StringVar(value='a4')
        paper_sizes = ['a4', 'a4_half_portrait', 'a0_portrait', 'a0_landscape', '16:9_monitor', 'custom']
        self.paper_size_combo = ttk.Combobox(control_panel, textvariable=self.paper_size_var, values=paper_sizes)
        self.paper_size_combo.pack(fill="x", pady=(0, 10))
        self.paper_size_combo.bind("<<ComboboxSelected>>", self._on_paper_size_change)

        # Frame for custom size entries
        self.custom_size_frame = ttk.Frame(control_panel)
        self.custom_size_frame.pack(fill='x')

        ttk.Label(self.custom_size_frame, text="Width (cm)").grid(row=0, column=0, sticky='w')
        self.custom_width_var = tk.DoubleVar(value=21.0)
        ttk.Entry(self.custom_size_frame, textvariable=self.custom_width_var).grid(row=0, column=1, padx=5)

        ttk.Label(self.custom_size_frame, text="Height (cm)").grid(row=1, column=0, sticky='w')
        self.custom_height_var = tk.DoubleVar(value=29.7)
        ttk.Entry(self.custom_size_frame, textvariable=self.custom_height_var).grid(row=1, column=1, padx=5)
        
        # Add a button to apply the new paper size
        self.update_paper_button = ttk.Button(control_panel, text="Update Paper Size", command=self.update_paper_size)
        self.update_paper_button.pack(pady=(10, 0), fill="x")


        # Right side: Control Panel : Style controls
        style_panel = ttk.LabelFrame(right_column_frame, text="Style", padding=10)
        style_panel.pack(fill="x")
        # style_panel.grid(row=1, column=1, sticky="n", padx=100, pady=50)

        # 1. Stylesheet Dropdown
        ttk.Label(style_panel, text="Stylesheet").pack(anchor="w")
        self.stylesheet_var = tk.StringVar(value="default")
        ttk.Combobox(style_panel, textvariable=self.stylesheet_var, values=["default", "modern",
                                                                            "nature-reviews", "economist"]).pack(
            fill="x", pady=(0, 10))

        # 2. Font Dropdown
        ttk.Label(style_panel, text="Font").pack(anchor="w")
        self.font_var = tk.StringVar(value="Helvetica")
        ttk.Combobox(style_panel, textvariable=self.font_var, values=["Helvetica", "Computer Modern Sans Serif", "Comic Sans MS"]).pack(
            fill="x", pady=(0, 10))

        # 3. Font Size (float entry)
        ttk.Label(style_panel, text="Font size").pack(anchor="w")
        self.font_size_var = tk.DoubleVar(value=11.0)
        ttk.Entry(style_panel, textvariable=self.font_size_var).pack(fill="x", pady=(0, 10))

        # 4. Tick mark font size
        ttk.Label(style_panel, text="Tick mark font size").pack(anchor="w")
        self.tick_font_size_var = tk.DoubleVar(value=9.0)
        ttk.Entry(style_panel, textvariable=self.tick_font_size_var).pack(fill="x")

        # 5. Capital or small letters
        self.use_capital_letters_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(style_panel, text="Use Capital Letters", variable=self.use_capital_letters_var,
                        command=self._update_panel_labels).pack(anchor="w", pady=(0, 10))

        ########################### MAKE THE PAPER PAGE ######################################
        self.panels = []  # set empty panels for now 
        # Create an empty rectangle first
        self.paper_rect = self.canvas.create_rectangle(0, 0, 0, 0, fill="white", outline="black", width=2)

        # Set the initial state and draw the paper
        self.update_paper_size()
        self._on_paper_size_change()

        """
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

        # Draw grid layout
        self.draw_grid((offset_x, offset_y, offset_x + paper_width_px, offset_y + paper_height_px),
                       self.grid_rows, self.grid_cols)
        """


    """
    def add_panel(self):
        label = chr(ord('A') + len(self.panels))  # A, B, C, ...
        panel = ResizablePanel(self.canvas, 50, 50, 200, 150, label=label)
        self.panels.append(panel)
    """

    """
    def add_panel(self):
        label = chr(ord('A') + len(self.panels))  # A, B, C, ...

        row, col = 0, 0
        rowspan, colspan = 2, 2  # Default size

        panel = GridPanel(
            self.canvas, self.paper_rect, self.grid_rows, self.grid_cols,
            row, col, rowspan, colspan, label=label
        )
        self.panels.append(panel)
    """

    def add_panel(self):

        # Determine the base character ('A' or 'a') from the checkbutton state
        if self.use_capital_letters_var.get():
            base_char = 'A'
        else:
            base_char = 'a'

        label = chr(ord(base_char) + len(self.panels))
        x0, y0, x1, y1 = self.canvas.coords(self.paper_rect)
        panel = ResizablePanel(self.canvas, x0 + 50, y0 + 50, 200, 150,
                               label=label,
                               grid_rows=self.grid_rows,
                               grid_cols=self.grid_cols,
                               paper_bbox=self.canvas.coords(self.paper_rect))
        self.panels.append(panel)

    def _update_panel_labels(self):
        """Updates the lettering case for all existing panels."""
        if self.use_capital_letters_var.get():
            base_char = 'A'
        else:
            base_char = 'a'

        for i, panel in enumerate(self.panels):
            new_label = chr(ord(base_char) + i)
            # Update the panel's internal text property
            panel.label_text = new_label
            # Update the text displayed on the canvas
            self.canvas.itemconfig(panel.label_id, text=new_label)


    def save_layout(self):
        """Gathers the current layout state and saves it to a JSON file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Layout File",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not filepath:
            return  # User cancelled

        # 1. Gather all the data into a dictionary
        layout_data = {
            'grid_settings': {
                'rows': self.grid_rows_var.get(),
                'cols': self.grid_cols_var.get()
            },
            'paper_settings': {
                'size_name': self.paper_size_var.get(),
                'custom_width_cm': self.custom_width_var.get(),
                'custom_height_cm': self.custom_height_var.get()
            },
            'panels': []
        }

        for panel in self.panels:
            panel_info = {
                'label': panel.label_text,
                'bbox': panel.get_bbox(),  # [x0, y0, x1, y1]
                'filepath': getattr(panel, 'filepath', None)
            }
            layout_data['panels'].append(panel_info)

        # 2. Write the data to the JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(layout_data, f, indent=4)
            print(f"Layout saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving layout: {e}")

    def load_layout(self):
        """Loads a layout state from a JSON file."""
        filepath = filedialog.askopenfilename(
            title="Open Layout File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not filepath:
            return  # User cancelled

        try:
            with open(filepath, 'r') as f:
                layout_data = json.load(f)

            # 1. Clear the current state
            for panel in self.panels:
                self.canvas.delete(panel.rect)
                self.canvas.delete(panel.label_id)
                if panel.file_label_id:
                    self.canvas.delete(panel.file_label_id)
            self.panels.clear()

            # 2. Apply global settings from the file
            self.grid_rows_var.set(layout_data['grid_settings']['rows'])
            self.grid_cols_var.set(layout_data['grid_settings']['cols'])
            self.paper_size_var.set(layout_data['paper_settings']['size_name'])
            self.custom_width_var.set(layout_data['paper_settings']['custom_width_cm'])
            self.custom_height_var.set(layout_data['paper_settings']['custom_height_cm'])

            # Update the canvas to reflect these settings
            self.update_paper_size()

            # 3. Recreate the panels
            for panel_info in layout_data['panels']:
                bbox = panel_info['bbox']
                x0, y0, x1, y1 = bbox
                w, h = x1 - x0, y1 - y0

                panel = ResizablePanel(self.canvas, x0, y0, w, h,
                                       label=panel_info['label'],
                                       grid_rows=self.grid_rows,
                                       grid_cols=self.grid_cols,
                                       paper_bbox=self.canvas.coords(self.paper_rect))

                if panel_info.get('filepath'):
                    panel.display_file_path(panel_info['filepath'])

                self.panels.append(panel)

            print(f"Layout loaded successfully from {filepath}")

        except Exception as e:
            print(f"Error loading layout: {e}")


    def _browse_save_path(self):
        """Opens a file dialog to choose a save location and base filename."""
        initial_path = Path(self.save_path_var.get())

        filepath = filedialog.asksaveasfilename(
            initialdir=initial_path.parent,
            initialfile=initial_path.stem,
            title="Choose save location and base filename",
            filetypes=[("PDF file", "*.pdf"), ("SVG file", "*.svg"), ("All files", "*.*")]
        )

        if filepath:
            # We strip any extension the user provides, as we'll be adding .pdf and .svg ourselves
            p = Path(filepath)
            self.save_path_var.set(str(p.with_suffix('')))

    def draw_grid(self, paper_coords, nrows, ncols):
        x0, y0, x1, y1 = paper_coords
        paper_width = x1 - x0
        paper_height = y1 - y0

        col_width = paper_width / ncols
        row_height = paper_height / nrows

        # Draw vertical lines
        for i in range(1, ncols):
            x = x0 + i * col_width
            self.canvas.create_line(x, y0, x, y1, fill="lightgray", dash=(2, 2), tags="grid")

        # Draw horizontal lines
        for j in range(1, nrows):
            y = y0 + j * row_height
            self.canvas.create_line(x0, y, x1, y, fill="lightgray", dash=(2, 2), tags="grid")

    def update_grid(self):
        try:
            rows = self.grid_rows_var.get()
            cols = self.grid_cols_var.get()

            if rows < 1 or cols < 1:
                raise ValueError

            self.grid_rows = rows
            self.grid_cols = cols

            # Clear previous grid lines
            self.canvas.delete("grid")

            # Redraw grid on paper rectangle
            self.draw_grid(self.canvas.coords(self.paper_rect), self.grid_rows, self.grid_cols)

            # Optionally: update existing panels to snap to the new grid
            for panel in self.panels:
                 # 1. Update the panel's internal knowledge of the grid
                panel.nrows = self.grid_rows
                panel.ncols = self.grid_cols
                panel.paper_bbox = self.canvas.coords(self.paper_rect)

                # 2. Get current position and calculate the new snapped position
                x0, y0, x1, y1 = panel.get_bbox()
                nx0, ny0 = panel.snap_to_grid(x0, y0)
                nx1, ny1 = panel.snap_to_grid(x1, y1)

                # 3. Move the existing panel items to the new snapped coordinates
                panel.canvas.coords(panel.rect, nx0, ny0, nx1, ny1)
                panel.canvas.coords(panel.label_id, nx0 + 6, ny0 + 6)
                if panel.file_label_id:
                    cx = (nx0 + nx1) / 2
                    cy = (ny0 + ny1) / 2
                    panel.canvas.coords(panel.file_label_id, cx, cy)
                
                # panel.canvas.delete(panel.rect)
                # panel.canvas.delete(panel.label_id)
                # panel.rect, panel.label_id = panel._draw_panel()
                # panel._bind_events()

        except Exception as e:
            print("Invalid grid dimensions:", e)
    
    def _on_paper_size_change(self, event=None):
        """Enables or disables the custom size entry fields based on dropdown selection."""
        if self.paper_size_var.get() == 'custom':
            for child in self.custom_size_frame.winfo_children():
                child.configure(state='normal')
        else:
            for child in self.custom_size_frame.winfo_children():
                child.configure(state='disabled')

    def update_paper_size(self):
        """Resizes the paper rectangle on the canvas based on the selected size."""
        # Dictionary of preset dimensions in cm (width, height)
        PAPER_DIMENSIONS = {
            'a4': (21.0, 29.7),
            'a4_half_portrait': (10.5, 29.7),
            'a0_portrait': (84.1, 118.9),
            'a0_landscape': (118.9, 84.1),
            '16:9_monitor': (59.7, 33.6),  # For a 27-inch monitor
        }

        selection = self.paper_size_var.get()

        if selection == 'custom':
            paper_width_cm = self.custom_width_var.get()
            paper_height_cm = self.custom_height_var.get()
        else:
            paper_width_cm, paper_height_cm = PAPER_DIMENSIONS[selection]
            self.custom_width_var.set(paper_width_cm)
            self.custom_height_var.set(paper_height_cm)
        
        # Store the true dimensions for the final figure export
        self.paper_width_cm = paper_width_cm
        self.paper_height_cm = paper_height_cm

        # --- SCALING LOGIC ---
        # 1. Calculate the paper's true pixel size
        true_width_px = cm_to_px(self.paper_width_cm, self.dpi)
        true_height_px = cm_to_px(self.paper_height_cm, self.dpi)

        # 2. Define the available drawing area on the canvas
        offset_x = 50
        offset_y = 50
        canvas_area_w = self.canvas.winfo_width() - (2 * offset_x)
        canvas_area_h = self.canvas.winfo_height() - (2 * offset_y)
        
        if canvas_area_w <= 1: # Fallback if canvas not drawn yet
            canvas_area_w = 800 - (2 * offset_x)
            canvas_area_h = 1300 - (2 * offset_y)
        
        # 3. Calculate the scale factor to fit the paper in the area
        scale = 1.0
        if true_width_px > canvas_area_w or true_height_px > canvas_area_h:
            scale_w = canvas_area_w / true_width_px
            scale_h = canvas_area_h / true_height_px
            scale = min(scale_w, scale_h)

        # 4. Calculate the scaled display size
        display_width_px = true_width_px * scale
        display_height_px = true_height_px * scale
        
        # 5. Resize the paper rectangle on the canvas using the display size
        self.canvas.coords(
            self.paper_rect,
            offset_x,
            offset_y,
            offset_x + display_width_px,
            offset_y + display_height_px
        )

        # Update the grid to match the new size
        self.update_grid()
        
        # Also update the paper_bbox for all existing panels
        for panel in self.panels:
            panel.paper_bbox = self.canvas.coords(self.paper_rect)


    def make_figures(self):

        # Use the stored true dimensions (in inches) for the final figure, not the canvas preview size.
        fig_width_in = self.paper_width_cm / 2.54
        fig_height_in = self.paper_height_cm / 2.54

        output_dpi = 300

        style_name = self.stylesheet_var.get()
        selected_font = self.font_var.get()  # Get the font from the dropdown
        rc_params = {
            'pdf.fonttype': 42,
            'font.family': selected_font
        }

        with plt.style.context(splstyle.get_style(style_name)):
            with plt.rc_context(rc=rc_params):
                fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=output_dpi)

                # Use GridSpec for layout
                from matplotlib.gridspec import GridSpec

                grid_rows = self.grid_rows_var.get()
                grid_cols = self.grid_cols_var.get()
                gs = GridSpec(grid_rows, grid_cols, figure=fig)

                # Get paper bounding box
                paper_x0, paper_y0, paper_x1, paper_y1 = self.canvas.coords(self.paper_rect)
                paper_w = paper_x1 - paper_x0
                paper_h = paper_y1 - paper_y0

                for panel in self.panels:
                    x0, y0, x1, y1 = panel.get_bbox()
                    rel_x0 = (x0 - paper_x0) / paper_w
                    rel_y0 = (y0 - paper_y0) / paper_h
                    rel_x1 = (x1 - paper_x0) / paper_w
                    rel_y1 = (y1 - paper_y0) / paper_h

                    # Convert relative coords into grid rows/cols
                    col0 = int(round(rel_x0 * grid_cols))
                    row0 = int(round(rel_y0 * grid_rows))
                    col1 = int(round(rel_x1 * grid_cols))
                    row1 = int(round(rel_y1 * grid_rows))

                    # Clamp to bounds
                    col0, col1 = sorted([max(0, min(col0, grid_cols - 1)), max(0, min(col1, grid_cols))])
                    row0, row1 = sorted([max(0, min(row0, grid_rows - 1)), max(0, min(row1, grid_rows))])

                    if col0 == col1:
                        col1 += 1
                    if row0 == row1:
                        row1 += 1

                    subfig_ax = fig.add_subplot(gs[row0:row1, col0:col1])

                    # Add sub figure lettering
                    subfig_ax.text(-0.1, 1.05, panel.label_text, transform=subfig_ax.transAxes,
                                   fontsize=14, fontweight='bold', va='bottom', ha='right')

                    subfig_ax.set_xticks([])
                    subfig_ax.set_yticks([])

                    filepath = getattr(panel, "filepath", None)
                    if filepath:
                        suffix = Path(filepath).suffix.lower()
                        if suffix in [".png", ".jpg", ".jpeg", ".svg"]:
                            img = mpimg.imread(filepath)
                            subfig_ax.imshow(img)
                            subfig_ax.axis("off")
                        elif suffix == ".pkl":
                            try:
                                with open(filepath, "rb") as f:
                                    original_fig = pickle.load(f)

                                buf = io.BytesIO()
                                pickle.dump(original_fig, buf)
                                buf.seek(0)
                                fig_copy = pickle.load(buf)

                                source_axes = fig_copy.get_axes()
                                if len(source_axes) == 1:
                                    copy_axes_content(source_axes[0], subfig_ax)
                                else:
                                    subrows = 1
                                    subcols = len(source_axes)
                                    subfig_ax.axis('off')
                                    inner_gs = subfig_ax.get_subplotspec().subgridspec(subrows, subcols, wspace=0.3)

                                    for i, src_ax in enumerate(source_axes):
                                        sub_ax = fig.add_subplot(inner_gs[0, i])
                                        copy_axes_content(src_ax, sub_ax)

                            except Exception as e:
                                subfig_ax.text(0.5, 0.5, f"Failed to load:\n{Path(filepath).name}", ha="center", va="center")
                    else:
                        subfig_ax.set_facecolor("#f0f0f0")
                        subfig_ax.text(0.5, 0.5, "Empty", ha="center", va="center")

                fig.suptitle("Composed Layout", fontsize=14)
                fig.tight_layout()
                # fig.show()

                # --- SAVE THE FIGURE ---
                save_path_str = self.save_path_var.get()
                if save_path_str:
                    try:
                        p = Path(save_path_str)
                        # Ensure the parent directory exists
                        p.parent.mkdir(parents=True, exist_ok=True)

                        # Define paths for both file types
                        pdf_path = p.with_suffix('.pdf')
                        svg_path = p.with_suffix('.svg')

                        # Save as PDF and SVG
                        fig.savefig(pdf_path)
                        fig.savefig(svg_path, transparent=True)

                        print(f"Figure saved successfully to:\n  {pdf_path}\n  {svg_path}")

                    except Exception as e:
                        print(f"Error saving figure: {e}")

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

    # also save as png
    fig.savefig(save_path[0:-4])

    # Example figure 2 : imshow
    fig, ax = plt.subplots()
    with cbook.get_sample_data('grace_hopper.jpg') as image_file:
        image = plt.imread(image_file)

    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    save_name = 'example_image.pkl'
    if save_folder is not None:
        save_path = os.path.join(save_folder, save_name)
    else:
        save_path = save_name

    with open(save_path, 'wb') as f:  # should be 'wb' rather than 'w'
        pickle.dump(fig, f)

    # also save as png
    fig.savefig(save_path[0:-4])

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

    # also save as png
    fig.savefig(save_path[0:-4])


class GridPanel:
    def __init__(self, canvas, paper_rect_id, nrows, ncols, row, col, rowspan, colspan, label="A"):
        self.canvas = canvas
        self.nrows = nrows
        self.ncols = ncols
        self.row = row
        self.col = col
        self.rowspan = rowspan
        self.colspan = colspan
        self.label = label
        self.filepath = None

        # Convert to pixel coordinates
        self.paper_bbox = self.canvas.coords(paper_rect_id)
        self.rect, self.label_id = self._draw_panel()

        # Draffing of the grid panel
        self.canvas.tag_bind(self.rect, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.rect, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.label_id, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.label_id, "<B1-Motion>", self.on_drag)
        self.drag_data = {"x": 0, "y": 0}

        self.canvas.tag_bind(self.rect, "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind(self.label_id, "<ButtonRelease-1>", self.on_release)

    def _draw_panel(self):
        x0, y0, x1, y1 = self.paper_bbox
        w = (x1 - x0) / self.ncols
        h = (y1 - y0) / self.nrows

        px0 = x0 + self.col * w
        py0 = y0 + self.row * h
        px1 = px0 + self.colspan * w
        py1 = py0 + self.rowspan * h

        rect_id = self.canvas.create_rectangle(px0, py0, px1, py1, fill="lightgray")
        label_id = self.canvas.create_text(px0 + 6, py0 + 6, anchor="nw", text=self.label, font=("Helvetica", 12, "bold"))
        return rect_id, label_id

    def get_gridspec_slice(self):
        return slice(self.row, self.row + self.rowspan), slice(self.col, self.col + self.colspan)

    def on_press(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag(self, event):
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]

        # Move rectangle visually
        self.canvas.move(self.rect, dx, dy)
        self.canvas.move(self.label_id, dx, dy)

        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_release(self, event):
        # Snap to grid
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        grid_x0, grid_y0, _, _ = self.paper_bbox
        w = (self.paper_bbox[2] - grid_x0) / self.ncols
        h = (self.paper_bbox[3] - grid_y0) / self.nrows

        # New grid coordinates (from snapped top-left)
        col = int((x0 - grid_x0 + w / 2) // w)
        row = int((y0 - grid_y0 + h / 2) // h)

        # Keep within bounds
        col = max(0, min(self.ncols - self.colspan, col))
        row = max(0, min(self.nrows - self.rowspan, row))

        self.row = row
        self.col = col

        # Redraw panel
        self.canvas.delete(self.rect)
        self.canvas.delete(self.label_id)
        self.rect, self.label_id = self._draw_panel()
        self._bind_events()

    def _bind_events(self):
        self.canvas.tag_bind(self.rect, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.rect, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.rect, "<ButtonRelease-1>", self.on_release)

        self.canvas.tag_bind(self.label_id, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.label_id, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.label_id, "<ButtonRelease-1>", self.on_release)


def copy_axes_content(src_ax, dest_ax):
    """
       Copies data artists (lines, patches, etc.) and properties from a source
       matplotlib axes to a destination axes. This version creates new artist
       instances to avoid reuse errors.
       """
    # --- 1. COPY DATA ARTISTS BY RECREATING THEM ---

    # Lines (e.g., from plot)
    for line in src_ax.lines:
        new_line = Line2D(
            xdata=line.get_xdata(),
            ydata=line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            label=line.get_label()
        )
        dest_ax.add_line(new_line)

    # Collections (e.g., from scatter)
    for collection in src_ax.collections:
        if isinstance(collection, PathCollection):
            # Deconstruct the original collection to get its core properties
            offsets = collection.get_offsets()

            # This is the key change: use the high-level scatter function
            # which is much more reliable than recreating the artist manually.
            dest_ax.scatter(
                offsets[:, 0],  # x-data
                offsets[:, 1],  # y-data
                s=collection.get_sizes(),
                c=collection.get_facecolors(),
                marker=collection.get_paths()[0] if collection.get_paths() else 'o',
                alpha=collection.get_alpha(),
                linewidths=collection.get_linewidths(),
                edgecolors=collection.get_edgecolors(),
                label=collection.get_label()
            )
            # If the original used a colormap, re-apply it
            if collection.get_array() is not None:
                new_collection = dest_ax.collections[-1]  # Get the collection we just made
                new_collection.set_array(collection.get_array())
                new_collection.set_cmap(collection.get_cmap())
                new_collection.set_norm(collection.get_norm())

    # Patches (e.g., from bar, hist)
    for patch in src_ax.patches:
        # Recreate the patch from its properties to break the link to the old figure
        if isinstance(patch, Rectangle):
            new_patch = Rectangle(
                xy=patch.get_xy(),
                width=patch.get_width(),
                height=patch.get_height(),
                angle=patch.get_angle(),
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth(),
                linestyle=patch.get_linestyle(),
                alpha=patch.get_alpha(),
                label=patch.get_label(),
            )
            dest_ax.add_patch(new_patch)
        else:
            # Fallback for other patch types if necessary
            new_patch = copy(patch)
            dest_ax.add_patch(new_patch)

    # Images (e.g., from imshow)
    for image in src_ax.images:

        new_image = dest_ax.imshow(
            image.get_array(),
            extent=image.get_extent(),
            cmap=image.get_cmap(),
            norm=image.norm,
            origin=getattr(image, '_origin', 'upper'),
            interpolation=image.get_interpolation()
        )
        new_image.set_clim(image.get_clim())

    # Text (excluding axis labels and title)
    for text in src_ax.texts:
        dest_ax.text(
            x=text.get_position()[0],
            y=text.get_position()[1],
            s=text.get_text(),
            transform=dest_ax.transData,  # Ensure text is in data coordinates
            ha=text.get_ha(),
            va=text.get_va(),
            fontsize=text.get_fontsize(),
            color=text.get_color()
        )

    # Axis limits and labels
    dest_ax.set_xlim(src_ax.get_xlim())
    dest_ax.set_ylim(src_ax.get_ylim())
    dest_ax.set_xscale(src_ax.get_xscale())
    dest_ax.set_yscale(src_ax.get_yscale())
    dest_ax.set_aspect(src_ax.get_aspect(), adjustable=src_ax.get_adjustable(), anchor=src_ax.get_anchor())

    dest_ax.set_title(src_ax.get_title())
    dest_ax.set_xlabel(src_ax.get_xlabel())
    dest_ax.set_ylabel(src_ax.get_ylabel())

    # Copy ticks, tick labels, and grid status
    dest_ax.set_xticks(src_ax.get_xticks())
    dest_ax.set_xticklabels([label.get_text() for label in src_ax.get_xticklabels()])
    dest_ax.set_yticks(src_ax.get_yticks())
    dest_ax.set_yticklabels([label.get_text() for label in src_ax.get_yticklabels()])
    dest_ax.grid(src_ax.get_xgridlines()[0].get_visible() if src_ax.get_xgridlines() else False)

@app.command()
def main():

    logger.info('Running layout')

    # TODO: ask the user which process to run : make_layout


if __name__ == "__main__":
    app()