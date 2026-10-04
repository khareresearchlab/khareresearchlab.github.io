#!/usr/bin/env python3
"""
HDF5 Free Editor
================

A general-purpose desktop GUI for inspecting and editing HDF5 files safely.

Key design choice:
- The original file is never edited directly.
- When a file is opened, the program creates a temporary working copy.
- Use Save or Save As to write the edited copy back to disk.

Supported operations:
- Browse groups and datasets
- Preview and edit scalar, 1-D, and 2-D dataset slices
- View N-D datasets through user-entered slices that resolve to <= 2 dimensions
- Delete leading/trailing rows or any index range along any axis
- Resize datasets while preserving overlapping data
- Replace a dataset from CSV or NPY
- Export datasets to CSV or NPY
- Add, update, and delete HDF5 attributes
- Create groups and datasets
- Rename or delete groups/datasets

Requirements:
    pip install PySide6 h5py numpy

Run:
    python hdf5_free_editor.py
"""

from __future__ import annotations

import ast
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import h5py
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "HDF5 Free Editor"
MAX_PREVIEW_CELLS_DEFAULT = 50_000


def human_shape(shape: Sequence[int]) -> str:
    return "scalar" if len(shape) == 0 else " × ".join(str(x) for x in shape)


def parse_shape(text: str) -> Tuple[int, ...]:
    text = text.strip()
    if not text or text.lower() in {"scalar", "()"}:
        return ()
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    shape = tuple(int(p) for p in parts)
    if any(n < 0 for n in shape):
        raise ValueError("Shape dimensions cannot be negative.")
    return shape


def split_slice_tokens(expr: str) -> list[str]:
    """Split a simple HDF5/Numpy slice expression on top-level commas."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in expr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tokens.append("".join(current).strip())
    return tokens


def parse_slice_component(token: str):
    token = token.strip()
    if token in {"", ":"}:
        return slice(None)
    if token == "...":
        return Ellipsis
    if ":" in token:
        pieces = token.split(":")
        if len(pieces) > 3:
            raise ValueError(f"Invalid slice component: {token!r}")
        vals = []
        for piece in pieces:
            piece = piece.strip()
            vals.append(None if piece == "" else int(piece))
        while len(vals) < 3:
            vals.append(None)
        return slice(vals[0], vals[1], vals[2])
    return int(token)


def parse_slice_expression(expr: str, ndim: int) -> tuple:
    expr = expr.strip()
    if ndim == 0:
        return ()
    if not expr:
        return tuple(slice(None) for _ in range(ndim))

    parts = [parse_slice_component(t) for t in split_slice_tokens(expr)]
    ellipsis_count = sum(1 for x in parts if x is Ellipsis)
    if ellipsis_count > 1:
        raise ValueError("Only one ellipsis (...) is allowed.")

    if ellipsis_count == 1:
        idx = parts.index(Ellipsis)
        explicit = len(parts) - 1
        fill = ndim - explicit
        if fill < 0:
            raise ValueError("Too many indices for this dataset.")
        parts = parts[:idx] + [slice(None)] * fill + parts[idx + 1 :]

    if len(parts) < ndim:
        parts.extend([slice(None)] * (ndim - len(parts)))
    if len(parts) > ndim:
        raise ValueError(f"Dataset has {ndim} dimensions, but {len(parts)} indices were supplied.")
    return tuple(parts)


def default_slice_for_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return ""
    return ", ".join(":" for _ in shape)


def jsonish_to_value(text: str) -> Any:
    """Interpret JSON/Python-like literals; otherwise keep the input as text."""
    stripped = text.strip()
    if stripped == "":
        return ""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except Exception:
            pass
    return text


def value_to_display(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=1000)
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def convert_cell_text(text: str, dtype: np.dtype) -> Any:
    """Convert table text into a value compatible with an HDF5 dataset dtype."""
    if dtype.fields:
        raise TypeError("Compound/structured dtypes are preview-only in this editor.")

    kind = dtype.kind
    text = text.strip()

    if kind in {"S", "U", "O"}:
        return text
    if kind == "b":
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot interpret {text!r} as boolean.")
    if kind in {"i", "u"}:
        return int(text, 0)
    if kind == "f":
        return float(text)
    if kind == "c":
        return complex(text)
    if kind in {"m", "M"}:
        return np.array(text, dtype=dtype).item()

    return np.array(text, dtype=dtype).item()


def safe_dataset_create_kwargs(ds: h5py.Dataset, new_shape: tuple[int, ...]) -> dict:
    """Carry over storage options that remain valid for a recreated dataset."""
    kwargs: dict[str, Any] = {"dtype": ds.dtype}

    if ds.compression is not None:
        kwargs["compression"] = ds.compression
        if ds.compression_opts is not None:
            kwargs["compression_opts"] = ds.compression_opts
    if ds.shuffle:
        kwargs["shuffle"] = True
    if ds.fletcher32:
        kwargs["fletcher32"] = True
    if ds.scaleoffset is not None:
        kwargs["scaleoffset"] = ds.scaleoffset

    # Preserve chunking when possible, but clamp chunks to nonzero dimensions.
    if ds.chunks is not None and len(ds.chunks) == len(new_shape) and all(n > 0 for n in new_shape):
        kwargs["chunks"] = tuple(max(1, min(int(c), int(n))) for c, n in zip(ds.chunks, new_shape))

    fillvalue = ds.fillvalue
    if fillvalue is not None:
        kwargs["fillvalue"] = fillvalue

    return kwargs


class CreateDatasetDialog(QDialog):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Create dataset")

        form = QFormLayout(self)
        self.name_edit = QLineEdit()
        self.shape_edit = QLineEdit("100, 3")
        self.dtype_edit = QLineEdit("float64")
        self.fill_edit = QLineEdit("0")

        form.addRow("Name:", self.name_edit)
        form.addRow("Shape:", self.shape_edit)
        form.addRow("NumPy dtype:", self.dtype_edit)
        form.addRow("Fill value:", self.fill_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[str, tuple[int, ...], np.dtype, Any]:
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("Dataset name is required.")
        shape = parse_shape(self.shape_edit.text())
        dtype = np.dtype(self.dtype_edit.text().strip())
        fill = convert_cell_text(self.fill_edit.text(), dtype)
        return name, shape, dtype, fill


class HDF5Editor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 850)

        self.original_path: Optional[str] = None
        self.working_path: Optional[str] = None
        self.h5: Optional[h5py.File] = None
        self.current_path: Optional[str] = None
        self.current_slice: tuple = ()
        self.current_view_shape: tuple[int, ...] = ()
        self.dirty = False

        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage("Open an HDF5 file to begin.")

    # ---------------- UI ----------------

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_to_original)
        menu.addAction(save_action)

        save_as_action = QAction("Save &As…", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_as)
        menu.addAction(save_as_action)

        menu.addSeparator()

        close_action = QAction("&Close file", self)
        close_action.triggered.connect(self.close_current_file)
        menu.addAction(close_action)

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        file_bar = QHBoxLayout()
        self.open_btn = QPushButton("Open HDF5")
        self.save_btn = QPushButton("Save")
        self.save_as_btn = QPushButton("Save As")
        self.refresh_btn = QPushButton("Refresh")
        self.file_label = QLabel("No file loaded")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.open_btn.clicked.connect(self.open_file)
        self.save_btn.clicked.connect(self.save_to_original)
        self.save_as_btn.clicked.connect(self.save_as)
        self.refresh_btn.clicked.connect(self.refresh_all)

        file_bar.addWidget(self.open_btn)
        file_bar.addWidget(self.save_btn)
        file_bar.addWidget(self.save_as_btn)
        file_bar.addWidget(self.refresh_btn)
        file_bar.addWidget(self.file_label, 1)
        root.addLayout(file_bar)

        splitter = QSplitter(Qt.Horizontal)

        # Left side: HDF5 tree
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("HDF5 structure"))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type / shape"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self.on_tree_selection)
        left_layout.addWidget(self.tree, 1)

        structure_buttons = QHBoxLayout()
        self.new_group_btn = QPushButton("New group")
        self.new_dataset_btn = QPushButton("New dataset")
        self.rename_btn = QPushButton("Rename")
        self.delete_btn = QPushButton("Delete")
        self.new_group_btn.clicked.connect(self.create_group)
        self.new_dataset_btn.clicked.connect(self.create_dataset)
        self.rename_btn.clicked.connect(self.rename_current)
        self.delete_btn.clicked.connect(self.delete_current)
        structure_buttons.addWidget(self.new_group_btn)
        structure_buttons.addWidget(self.new_dataset_btn)
        structure_buttons.addWidget(self.rename_btn)
        structure_buttons.addWidget(self.delete_btn)
        left_layout.addLayout(structure_buttons)

        splitter.addWidget(left)

        # Right side: tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._make_data_tab(), "Data")
        self.tabs.addTab(self._make_attributes_tab(), "Attributes")
        self.tabs.addTab(self._make_log_tab(), "Log")
        splitter.addWidget(self.tabs)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self._set_controls_enabled(False)

    def _make_data_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QHBoxLayout()
        self.path_label = QLabel("Path: —")
        self.shape_label = QLabel("Shape: —")
        self.dtype_label = QLabel("Dtype: —")
        info.addWidget(self.path_label, 2)
        info.addWidget(self.shape_label, 1)
        info.addWidget(self.dtype_label, 1)
        layout.addLayout(info)

        slice_bar = QHBoxLayout()
        slice_bar.addWidget(QLabel("Slice:"))
        self.slice_edit = QLineEdit()
        self.slice_edit.setPlaceholderText("Examples: :, :   or   68:, :   or   0, :, 5")
        self.preview_limit = QSpinBox()
        self.preview_limit.setRange(100, 2_000_000)
        self.preview_limit.setValue(MAX_PREVIEW_CELLS_DEFAULT)
        self.preview_limit.setSingleStep(10_000)
        self.load_slice_btn = QPushButton("Load slice")
        self.load_slice_btn.clicked.connect(self.load_current_slice)

        slice_bar.addWidget(self.slice_edit, 1)
        slice_bar.addWidget(QLabel("Max cells:"))
        slice_bar.addWidget(self.preview_limit)
        slice_bar.addWidget(self.load_slice_btn)
        layout.addLayout(slice_bar)

        self.data_table = QTableWidget()
        self.data_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.data_table, 1)

        edit_buttons = QHBoxLayout()
        self.apply_edits_btn = QPushButton("Apply cell edits")
        self.trim_leading_btn = QPushButton("Delete leading rows/indices")
        self.trim_trailing_btn = QPushButton("Delete trailing rows/indices")
        self.delete_range_btn = QPushButton("Delete index range")
        self.resize_btn = QPushButton("Resize dataset")

        self.apply_edits_btn.clicked.connect(self.apply_cell_edits)
        self.trim_leading_btn.clicked.connect(lambda: self.trim_dataset(leading=True))
        self.trim_trailing_btn.clicked.connect(lambda: self.trim_dataset(leading=False))
        self.delete_range_btn.clicked.connect(self.delete_index_range)
        self.resize_btn.clicked.connect(self.resize_dataset)

        edit_buttons.addWidget(self.apply_edits_btn)
        edit_buttons.addWidget(self.trim_leading_btn)
        edit_buttons.addWidget(self.trim_trailing_btn)
        edit_buttons.addWidget(self.delete_range_btn)
        edit_buttons.addWidget(self.resize_btn)
        layout.addLayout(edit_buttons)

        io_buttons = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_npy_btn = QPushButton("Export NPY")
        self.replace_csv_btn = QPushButton("Replace from CSV")
        self.replace_npy_btn = QPushButton("Replace from NPY")

        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_npy_btn.clicked.connect(self.export_npy)
        self.replace_csv_btn.clicked.connect(self.replace_from_csv)
        self.replace_npy_btn.clicked.connect(self.replace_from_npy)

        io_buttons.addWidget(self.export_csv_btn)
        io_buttons.addWidget(self.export_npy_btn)
        io_buttons.addWidget(self.replace_csv_btn)
        io_buttons.addWidget(self.replace_npy_btn)
        layout.addLayout(io_buttons)

        return tab

    def _make_attributes_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.attr_table = QTableWidget(0, 2)
        self.attr_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.attr_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attr_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.attr_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.attr_table, 1)

        buttons = QHBoxLayout()
        self.add_attr_btn = QPushButton("Add attribute")
        self.update_attr_btn = QPushButton("Apply attribute edits")
        self.delete_attr_btn = QPushButton("Delete selected attribute")

        self.add_attr_btn.clicked.connect(self.add_attribute)
        self.update_attr_btn.clicked.connect(self.apply_attribute_edits)
        self.delete_attr_btn.clicked.connect(self.delete_attribute)

        buttons.addWidget(self.add_attr_btn)
        buttons.addWidget(self.update_attr_btn)
        buttons.addWidget(self.delete_attr_btn)
        layout.addLayout(buttons)

        return tab

    def _make_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return tab

    # ---------------- File handling ----------------

    def _set_controls_enabled(self, enabled: bool):
        for widget in (
            self.save_btn,
            self.save_as_btn,
            self.refresh_btn,
            self.tree,
            self.new_group_btn,
            self.new_dataset_btn,
            self.rename_btn,
            self.delete_btn,
            self.slice_edit,
            self.preview_limit,
            self.load_slice_btn,
            self.apply_edits_btn,
            self.trim_leading_btn,
            self.trim_trailing_btn,
            self.delete_range_btn,
            self.resize_btn,
            self.export_csv_btn,
            self.export_npy_btn,
            self.replace_csv_btn,
            self.replace_npy_btn,
            self.attr_table,
            self.add_attr_btn,
            self.update_attr_btn,
            self.delete_attr_btn,
        ):
            widget.setEnabled(enabled)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 file",
            "",
            "HDF5 files (*.h5 *.hdf5 *.hdf);;All files (*)",
        )
        if not path:
            return

        self.close_current_file(quiet=True)

        suffix = Path(path).suffix or ".h5"
        fd, temp_path = tempfile.mkstemp(prefix="hdf5_editor_", suffix=suffix)
        os.close(fd)
        shutil.copy2(path, temp_path)

        try:
            self.h5 = h5py.File(temp_path, "r+")
        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

        self.original_path = os.path.abspath(path)
        self.working_path = temp_path
        self.dirty = False
        self.file_label.setText(f"Working copy of: {self.original_path}")
        self._set_controls_enabled(True)
        self.populate_tree()
        self.log(f"Opened {self.original_path}")
        self.log(f"Temporary working copy: {self.working_path}")
        self.statusBar().showMessage("File opened. Changes are being made to a temporary working copy.")

    def save_to_original(self):
        if not self.h5 or not self.original_path or not self.working_path:
            return

        answer = QMessageBox.question(
            self,
            "Overwrite original file?",
            f"This will replace the original file:\n\n{self.original_path}\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self.h5.flush()
        self._copy_working_file(self.original_path)
        self.dirty = False
        self.log(f"Saved working copy over original: {self.original_path}")
        self.statusBar().showMessage("Saved to original file.")

    def save_as(self):
        if not self.h5 or not self.working_path:
            return

        default_name = ""
        if self.original_path:
            p = Path(self.original_path)
            default_name = str(p.with_name(f"{p.stem}_edited{p.suffix}"))

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save edited HDF5 file as",
            default_name,
            "HDF5 files (*.h5 *.hdf5 *.hdf);;All files (*)",
        )
        if not path:
            return

        self.h5.flush()
        self._copy_working_file(path)
        self.dirty = False
        self.log(f"Saved edited file as: {path}")
        self.statusBar().showMessage(f"Saved as {path}")

    def _copy_working_file(self, destination: str):
        assert self.h5 is not None
        assert self.working_path is not None

        selected = self.current_path
        self.h5.flush()
        self.h5.close()
        self.h5 = None

        try:
            shutil.copy2(self.working_path, destination)
        finally:
            self.h5 = h5py.File(self.working_path, "r+")
            self.populate_tree(select_path=selected)

    def close_current_file(self, quiet: bool = False):
        if self.h5 is None and self.working_path is None:
            return

        if self.dirty and not quiet:
            answer = QMessageBox.question(
                self,
                "Discard unsaved edits?",
                "The working copy contains edits that have not been saved.\n\nDiscard them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            if self.h5 is not None:
                self.h5.close()
        finally:
            self.h5 = None

        if self.working_path:
            try:
                os.remove(self.working_path)
            except OSError:
                pass

        self.original_path = None
        self.working_path = None
        self.current_path = None
        self.current_slice = ()
        self.current_view_shape = ()
        self.dirty = False

        self.tree.clear()
        self.data_table.clear()
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
        self.attr_table.setRowCount(0)
        self.file_label.setText("No file loaded")
        self.path_label.setText("Path: —")
        self.shape_label.setText("Shape: —")
        self.dtype_label.setText("Dtype: —")
        self._set_controls_enabled(False)
        self.statusBar().showMessage("No file loaded.")

    def closeEvent(self, event):
        if self.dirty:
            answer = QMessageBox.question(
                self,
                "Exit without saving?",
                "The working copy contains unsaved edits.\n\nExit and discard them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        self.close_current_file(quiet=True)
        event.accept()

    # ---------------- Tree ----------------

    def populate_tree(self, select_path: Optional[str] = None):
        self.tree.clear()
        if self.h5 is None:
            return

        root_item = QTreeWidgetItem(["/", "Group"])
        root_item.setData(0, Qt.UserRole, "/")
        self.tree.addTopLevelItem(root_item)

        def add_children(group: h5py.Group, parent_item: QTreeWidgetItem):
            for name in sorted(group.keys()):
                obj = group[name]
                full_path = obj.name
                if isinstance(obj, h5py.Group):
                    item = QTreeWidgetItem([name, "Group"])
                    item.setData(0, Qt.UserRole, full_path)
                    parent_item.addChild(item)
                    add_children(obj, item)
                elif isinstance(obj, h5py.Dataset):
                    item = QTreeWidgetItem([name, f"Dataset {human_shape(obj.shape)}"])
                    item.setData(0, Qt.UserRole, full_path)
                    parent_item.addChild(item)

        add_children(self.h5["/"], root_item)
        root_item.setExpanded(True)

        target = select_path or "/"
        found = self._find_tree_item(root_item, target)
        if found:
            self.tree.setCurrentItem(found)
            found.setSelected(True)
            self.tree.scrollToItem(found)

    def _find_tree_item(self, item: QTreeWidgetItem, path: str) -> Optional[QTreeWidgetItem]:
        if item.data(0, Qt.UserRole) == path:
            return item
        for i in range(item.childCount()):
            found = self._find_tree_item(item.child(i), path)
            if found:
                return found
        return None

    def on_tree_selection(self):
        items = self.tree.selectedItems()
        if not items or self.h5 is None:
            return

        path = items[0].data(0, Qt.UserRole)
        if not path or path not in self.h5:
            return

        self.current_path = path
        obj = self.h5[path]
        self.path_label.setText(f"Path: {path}")
        self.refresh_attributes()

        if isinstance(obj, h5py.Dataset):
            self.shape_label.setText(f"Shape: {human_shape(obj.shape)}")
            self.dtype_label.setText(f"Dtype: {obj.dtype}")
            self.slice_edit.setText(default_slice_for_shape(obj.shape))
            self.load_current_slice()
            self.tabs.setCurrentIndex(0)
        else:
            self.shape_label.setText("Shape: Group")
            self.dtype_label.setText("Dtype: —")
            self.data_table.clear()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)

    def current_object(self):
        if self.h5 is None or self.current_path is None:
            return None
        if self.current_path not in self.h5:
            return None
        return self.h5[self.current_path]

    def current_dataset(self) -> Optional[h5py.Dataset]:
        obj = self.current_object()
        return obj if isinstance(obj, h5py.Dataset) else None

    # ---------------- Data viewing/editing ----------------

    def load_current_slice(self):
        ds = self.current_dataset()
        if ds is None:
            return

        try:
            selection = parse_slice_expression(self.slice_edit.text(), ds.ndim)
            arr = np.asarray(ds[selection])

            if arr.ndim > 2:
                raise ValueError(
                    f"The selected slice has {arr.ndim} dimensions. "
                    "Fix one or more dimensions with integer indices so the result is scalar, 1-D, or 2-D."
                )

            cells = int(arr.size) if arr.ndim else 1
            if cells > self.preview_limit.value():
                raise ValueError(
                    f"The selected slice contains {cells:,} cells, exceeding the preview limit "
                    f"of {self.preview_limit.value():,}. Narrow the slice or raise the limit."
                )

            self.current_slice = selection
            self.current_view_shape = arr.shape
            self._fill_data_table(arr, ds.dtype)
            self.statusBar().showMessage(f"Loaded slice {self.slice_edit.text() or '(scalar)'}")
        except Exception as exc:
            QMessageBox.critical(self, "Cannot load slice", str(exc))

    def _fill_data_table(self, arr: np.ndarray, dtype: np.dtype):
        self.data_table.blockSignals(True)
        try:
            self.data_table.clear()

            if arr.ndim == 0:
                self.data_table.setRowCount(1)
                self.data_table.setColumnCount(1)
                self.data_table.setHorizontalHeaderLabels(["Value"])
                value = arr.item()
                self.data_table.setItem(0, 0, QTableWidgetItem(value_to_display(value)))

            elif arr.ndim == 1:
                self.data_table.setRowCount(arr.shape[0])
                self.data_table.setColumnCount(1)
                self.data_table.setHorizontalHeaderLabels(["Value"])
                for r, value in enumerate(arr):
                    self.data_table.setItem(r, 0, QTableWidgetItem(value_to_display(value)))

            else:
                rows, cols = arr.shape
                self.data_table.setRowCount(rows)
                self.data_table.setColumnCount(cols)
                self.data_table.setHorizontalHeaderLabels([str(i) for i in range(cols)])
                for r in range(rows):
                    for c in range(cols):
                        self.data_table.setItem(r, c, QTableWidgetItem(value_to_display(arr[r, c])))

            editable = not bool(dtype.fields)
            triggers = (
                QAbstractItemView.DoubleClicked
                | QAbstractItemView.SelectedClicked
                | QAbstractItemView.EditKeyPressed
            )
            self.data_table.setEditTriggers(triggers if editable else QAbstractItemView.NoEditTriggers)
            self.apply_edits_btn.setEnabled(editable)
        finally:
            self.data_table.blockSignals(False)

    def apply_cell_edits(self):
        ds = self.current_dataset()
        if ds is None:
            return

        try:
            rows = self.data_table.rowCount()
            cols = self.data_table.columnCount()
            target_shape = self.current_view_shape

            if target_shape == ():
                value = convert_cell_text(self._table_text(0, 0), ds.dtype)
                new_data = np.array(value, dtype=ds.dtype)
            elif len(target_shape) == 1:
                vals = [convert_cell_text(self._table_text(r, 0), ds.dtype) for r in range(rows)]
                new_data = np.asarray(vals, dtype=ds.dtype)
            elif len(target_shape) == 2:
                vals = [
                    [convert_cell_text(self._table_text(r, c), ds.dtype) for c in range(cols)]
                    for r in range(rows)
                ]
                new_data = np.asarray(vals, dtype=ds.dtype)
            else:
                raise RuntimeError("Only scalar, 1-D, and 2-D views can be edited.")

            ds[self.current_slice] = new_data
            self._mark_dirty(f"Edited dataset cells: {ds.name}, slice={self.slice_edit.text()!r}")
            self.load_current_slice()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot apply edits", str(exc))

    def _table_text(self, row: int, col: int) -> str:
        item = self.data_table.item(row, col)
        return "" if item is None else item.text()

    # ---------------- Dataset structural edits ----------------

    def trim_dataset(self, leading: bool):
        ds = self.current_dataset()
        if ds is None or ds.ndim == 0:
            QMessageBox.information(self, "Not applicable", "Select a non-scalar dataset.")
            return

        axis, ok = QInputDialog.getInt(
            self,
            "Choose axis",
            f"Axis to trim (0 to {ds.ndim - 1}):",
            0,
            0,
            ds.ndim - 1,
        )
        if not ok:
            return

        count, ok = QInputDialog.getInt(
            self,
            "Number of indices",
            "How many leading indices should be deleted?" if leading
            else "How many trailing indices should be deleted?",
            1,
            1,
            max(1, ds.shape[axis]),
        )
        if not ok:
            return

        if count >= ds.shape[axis]:
            QMessageBox.warning(self, "Invalid trim", "The operation would remove the entire dataset.")
            return

        start = 0 if leading else ds.shape[axis] - count
        stop = count if leading else ds.shape[axis]
        self._delete_range_from_axis(axis, start, stop)

    def delete_index_range(self):
        ds = self.current_dataset()
        if ds is None or ds.ndim == 0:
            QMessageBox.information(self, "Not applicable", "Select a non-scalar dataset.")
            return

        axis, ok = QInputDialog.getInt(
            self,
            "Choose axis",
            f"Axis (0 to {ds.ndim - 1}):",
            0,
            0,
            ds.ndim - 1,
        )
        if not ok:
            return

        text, ok = QInputDialog.getText(
            self,
            "Delete index range",
            f"Enter a Python-style half-open range start:stop for axis {axis}\n"
            f"(axis length = {ds.shape[axis]}):",
            text="0:1",
        )
        if not ok:
            return

        try:
            pieces = text.split(":")
            if len(pieces) != 2:
                raise ValueError("Use start:stop, for example 0:68.")
            start = int(pieces[0].strip())
            stop = int(pieces[1].strip())
            self._delete_range_from_axis(axis, start, stop)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid range", str(exc))

    def _delete_range_from_axis(self, axis: int, start: int, stop: int):
        ds = self.current_dataset()
        if ds is None:
            return

        n = ds.shape[axis]
        start = start + n if start < 0 else start
        stop = stop + n if stop < 0 else stop
        start = max(0, min(start, n))
        stop = max(0, min(stop, n))

        if stop <= start:
            raise ValueError("The stop index must be greater than the start index.")
        if stop - start >= n:
            raise ValueError("The operation would remove the entire dataset.")

        slicer_before = [slice(None)] * ds.ndim
        slicer_after = [slice(None)] * ds.ndim
        slicer_before[axis] = slice(0, start)
        slicer_after[axis] = slice(stop, n)

        before = np.asarray(ds[tuple(slicer_before)])
        after = np.asarray(ds[tuple(slicer_after)])
        new_data = np.concatenate([before, after], axis=axis)

        path = ds.name
        self._replace_dataset(path, new_data, source_ds=ds)
        self.current_path = path
        self._mark_dirty(f"Deleted indices {start}:{stop} along axis {axis} from {path}")
        self.populate_tree(select_path=path)

    def resize_dataset(self):
        ds = self.current_dataset()
        if ds is None:
            return

        text, ok = QInputDialog.getText(
            self,
            "Resize dataset",
            "New shape, for example 1000, 20:",
            text=", ".join(map(str, ds.shape)) if ds.shape else "scalar",
        )
        if not ok:
            return

        try:
            old_shape = tuple(ds.shape)
            new_shape = parse_shape(text)
            if len(new_shape) != ds.ndim:
                raise ValueError(
                    f"Changing dimensionality is not supported here. "
                    f"Current ndim={ds.ndim}, requested ndim={len(new_shape)}."
                )

            old_data = np.asarray(ds[()])
            fill = ds.fillvalue
            if fill is None:
                if ds.dtype.kind in {"S", "U", "O"}:
                    fill = ""
                else:
                    fill = 0

            new_data = np.full(new_shape, fill, dtype=ds.dtype)
            overlap = tuple(slice(0, min(a, b)) for a, b in zip(ds.shape, new_shape))
            if ds.ndim == 0:
                new_data = np.asarray(old_data, dtype=ds.dtype)
            elif all(s.stop > 0 for s in overlap):
                new_data[overlap] = old_data[overlap]

            path = ds.name
            self._replace_dataset(path, new_data, source_ds=ds)
            self.current_path = path
            self._mark_dirty(f"Resized {path} from {old_shape} to {new_shape}")
            self.populate_tree(select_path=path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot resize dataset", str(exc))

    def _replace_dataset(self, path: str, data: np.ndarray, source_ds: Optional[h5py.Dataset] = None):
        if self.h5 is None:
            raise RuntimeError("No file is open.")

        if source_ds is None:
            source_ds = self.h5[path]

        attrs = {k: source_ds.attrs[k] for k in source_ds.attrs.keys()}
        kwargs = safe_dataset_create_kwargs(source_ds, tuple(data.shape))

        parent_path, name = path.rsplit("/", 1)
        parent_path = parent_path or "/"
        parent = self.h5[parent_path]

        del parent[name]
        new_ds = parent.create_dataset(name, data=data, **kwargs)
        for key, value in attrs.items():
            new_ds.attrs[key] = value
        self.h5.flush()

    # ---------------- Import/export ----------------

    def export_csv(self):
        ds = self.current_dataset()
        if ds is None:
            return
        data = np.asarray(ds[()])
        if data.ndim > 2:
            QMessageBox.information(
                self,
                "Cannot export as CSV",
                "CSV export supports scalar, 1-D, and 2-D datasets. Use NPY for higher dimensions.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export dataset as CSV", "", "CSV files (*.csv)")
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if data.ndim == 0:
                writer.writerow([value_to_display(data.item())])
            elif data.ndim == 1:
                for value in data:
                    writer.writerow([value_to_display(value)])
            else:
                for row in data:
                    writer.writerow([value_to_display(value) for value in row])

        self.log(f"Exported {ds.name} to CSV: {path}")

    def export_npy(self):
        ds = self.current_dataset()
        if ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export dataset as NPY", "", "NumPy files (*.npy)")
        if not path:
            return
        np.save(path, np.asarray(ds[()]))
        self.log(f"Exported {ds.name} to NPY: {path}")

    def replace_from_csv(self):
        ds = self.current_dataset()
        if ds is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Replace dataset from CSV", "", "CSV files (*.csv)")
        if not path:
            return

        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
            if raw.ndim == 0:
                raw = np.asarray(raw.item())
            converted = np.asarray(raw, dtype=ds.dtype)
            dataset_path = ds.name
            self._replace_dataset(dataset_path, converted, source_ds=ds)
            self.current_path = dataset_path
            self._mark_dirty(f"Replaced {dataset_path} from CSV: {path}")
            self.populate_tree(select_path=dataset_path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot replace from CSV", str(exc))

    def replace_from_npy(self):
        ds = self.current_dataset()
        if ds is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Replace dataset from NPY", "", "NumPy files (*.npy)")
        if not path:
            return

        try:
            data = np.load(path, allow_pickle=False)
            converted = np.asarray(data, dtype=ds.dtype)
            dataset_path = ds.name
            self._replace_dataset(dataset_path, converted, source_ds=ds)
            self.current_path = dataset_path
            self._mark_dirty(f"Replaced {dataset_path} from NPY: {path}")
            self.populate_tree(select_path=dataset_path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot replace from NPY", str(exc))

    # ---------------- Attributes ----------------

    def refresh_attributes(self):
        obj = self.current_object()
        self.attr_table.setRowCount(0)
        if obj is None:
            return

        for row, key in enumerate(sorted(obj.attrs.keys())):
            self.attr_table.insertRow(row)
            self.attr_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.attr_table.setItem(row, 1, QTableWidgetItem(value_to_display(obj.attrs[key])))

    def add_attribute(self):
        obj = self.current_object()
        if obj is None:
            return

        name, ok = QInputDialog.getText(self, "Add attribute", "Attribute name:")
        if not ok or not name.strip():
            return
        value, ok = QInputDialog.getMultiLineText(
            self,
            "Add attribute",
            "Value (JSON/Python literals are accepted):",
            "",
        )
        if not ok:
            return

        obj.attrs[name.strip()] = jsonish_to_value(value)
        self._mark_dirty(f"Added attribute {name.strip()!r} to {obj.name}")
        self.refresh_attributes()

    def apply_attribute_edits(self):
        obj = self.current_object()
        if obj is None:
            return

        try:
            table_values: dict[str, Any] = {}
            for row in range(self.attr_table.rowCount()):
                name_item = self.attr_table.item(row, 0)
                value_item = self.attr_table.item(row, 1)
                name = "" if name_item is None else name_item.text().strip()
                value_text = "" if value_item is None else value_item.text()
                if not name:
                    raise ValueError(f"Attribute row {row + 1} has an empty name.")
                table_values[name] = jsonish_to_value(value_text)

            for key in list(obj.attrs.keys()):
                if key not in table_values:
                    del obj.attrs[key]
            for key, value in table_values.items():
                obj.attrs[key] = value

            self._mark_dirty(f"Updated attributes for {obj.name}")
            self.refresh_attributes()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot apply attribute edits", str(exc))

    def delete_attribute(self):
        obj = self.current_object()
        rows = sorted({index.row() for index in self.attr_table.selectedIndexes()})
        if obj is None or not rows:
            return

        names = []
        for row in rows:
            item = self.attr_table.item(row, 0)
            if item is not None:
                names.append(item.text())

        for name in names:
            if name in obj.attrs:
                del obj.attrs[name]

        self._mark_dirty(f"Deleted attributes {names} from {obj.name}")
        self.refresh_attributes()

    # ---------------- Structure editing ----------------

    def selected_parent_group(self) -> Optional[h5py.Group]:
        obj = self.current_object()
        if obj is None or self.h5 is None:
            return None
        if isinstance(obj, h5py.Group):
            return obj
        return obj.parent

    def create_group(self):
        parent = self.selected_parent_group()
        if parent is None:
            return

        name, ok = QInputDialog.getText(self, "Create group", f"New group name inside {parent.name}:")
        if not ok or not name.strip():
            return

        try:
            group = parent.create_group(name.strip())
            self._mark_dirty(f"Created group {group.name}")
            self.populate_tree(select_path=group.name)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot create group", str(exc))

    def create_dataset(self):
        parent = self.selected_parent_group()
        if parent is None:
            return

        dialog = CreateDatasetDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        try:
            name, shape, dtype, fill = dialog.values()
            ds = parent.create_dataset(name, shape=shape, dtype=dtype, fillvalue=fill)
            if shape:
                ds[...] = fill
            else:
                ds[()] = fill
            self._mark_dirty(f"Created dataset {ds.name}, shape={shape}, dtype={dtype}")
            self.populate_tree(select_path=ds.name)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot create dataset", str(exc))

    def rename_current(self):
        obj = self.current_object()
        if obj is None or obj.name == "/":
            return

        old_path = obj.name
        old_name = old_path.rsplit("/", 1)[-1]
        new_name, ok = QInputDialog.getText(self, "Rename object", "New name:", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return

        try:
            parent_path = old_path.rsplit("/", 1)[0] or "/"
            new_path = parent_path.rstrip("/") + "/" + new_name.strip()
            self.h5.move(old_path, new_path)
            self.current_path = new_path
            self._mark_dirty(f"Renamed {old_path} to {new_path}")
            self.populate_tree(select_path=new_path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot rename object", str(exc))

    def delete_current(self):
        obj = self.current_object()
        if obj is None or obj.name == "/":
            return

        path = obj.name
        answer = QMessageBox.warning(
            self,
            "Delete HDF5 object?",
            f"Delete this object and all of its contents?\n\n{path}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            parent_path, name = path.rsplit("/", 1)
            parent = self.h5[parent_path or "/"]
            del parent[name]
            self.current_path = parent.name
            self._mark_dirty(f"Deleted {path}")
            self.populate_tree(select_path=parent.name)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot delete object", str(exc))

    # ---------------- General ----------------

    def refresh_all(self):
        if self.h5 is None:
            return
        selected = self.current_path
        self.h5.flush()
        self.populate_tree(select_path=selected)
        self.log("Refreshed file view.")

    def _mark_dirty(self, message: str):
        self.dirty = True
        if self.h5 is not None:
            self.h5.flush()
        self.log(message)
        self.statusBar().showMessage("Working copy modified. Use Save or Save As to keep the changes.")

    def log(self, message: str):
        self.log_box.append(message)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = HDF5Editor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
