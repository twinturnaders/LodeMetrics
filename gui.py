import sys
import os
import re
import asyncio
from collections import Counter
import webbrowser
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QFileDialog, QSplitter, QGroupBox, QLineEdit, QSpinBox,
    QStatusBar, QFormLayout, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QCursor, Qt

from db import LodeDB
from vision_engine import VisionEngine
from mercari_fetcher import MercariFetcher
from metadata_extractor import MetadataExtractor
from ocr_dialog import OCREditDialog

# --- Custom Drag-and-Drop Widget ---
class DropZone(QLabel):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #555;
                border-radius: 10px;
                background-color: #2b2b2b;
                color: #aaa;
                font-size: 16px;
            }
            QLabel:hover {
                border-color: #0078d7;
                background-color: #333;
                color: #fff;
            }
        """)
        self.setText("Drag & Drop an Image Here\nor Click to Browse")
        self.setAcceptDrops(True)
        self.file_path = None

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.file_dropped.emit(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            self.file_dropped.emit(file_path)

    def set_image(self, file_path: str):
        pixmap = QPixmap(file_path).scaled(280, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)
        self.file_path = file_path

# --- Background Workers ---
class PipelineWorker(QThread):
    progress = Signal(str, int)  # message, percentage
    finished = Signal()
    error = Signal(str)

    def __init__(self, db, vision_engine, keyword, max_pages):
        super().__init__()
        self.db = db
        self.vision_engine = vision_engine
        self.keyword = keyword
        self.max_pages = max_pages

    def run(self):
        try:
            self.progress.emit(f" Scraping '{self.keyword}'...", 10)
            fetcher = MercariFetcher(self.db)
            asyncio.run(fetcher.search_sold(self.keyword, max_pages=self.max_pages))
            
            self.progress.emit(f"📦 Scraped {self.max_pages * 100} listings", 40)
            
            self.progress.emit(f" Embedding photos...", 50)
            batch_size = self.max_pages * 100 
            self.vision_engine.process_batch(batch_size=batch_size)
            
            self.progress.emit(f"✅ Embedding complete", 90)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class AnalysisWorker(QThread):
    finished = Signal(list, dict, str)  # results, metadata, keyword_used
    progress = Signal(str, int)
    error = Signal(str)

    def __init__(self, vision_engine, metadata_extractor, image_path, keywords: str, top_k=8):
        super().__init__()
        self.vision_engine = vision_engine
        self.metadata_extractor = metadata_extractor
        self.image_path = image_path
        self.keywords = keywords
        self.top_k = top_k

    def run(self):
        try:
            self.progress.emit("🔍 Searching local database...", 20)
            
            # 1. LOCAL VISUAL SEARCH
            results = self.vision_engine.search_similar(self.image_path, top_k=20)
            
            self.progress.emit("📊 Calculating hybrid scores...", 30)
            
            metadata = {}
            needs_scrape = False
            scrape_keyword = self.keywords
            
            if results:
                # 2. HYBRID SCORING
                results = self.rerank_with_hybrid_score(results, self.keywords)
                
                # 3. CHECK IF WE NEED TO SCRAPE
                top_result = results[0]
                if top_result['similarity_score'] < 0.85:
                    needs_scrape = True
                    self.progress.emit(f"⚠️ Weak local match ({top_result['similarity_score']:.2f}). Need live data...", 40)
                else:
                    metadata = self.metadata_extractor.parse_comp_title(top_result['title'])
                    metadata['source'] = "Local DB Match"
                    self.progress.emit(f"✅ Found strong local match!", 100)
            else:
                needs_scrape = True
                self.progress.emit("📭 No local results. Scraping live data...", 40)
            
            if needs_scrape:
                # Return empty results with flag to trigger scrape
                self.finished.emit([], {"needs_scrape": True, "keyword": self.keywords}, self.keywords)
            else:
                self.finished.emit(results[:self.top_k], metadata, self.keywords)

        except Exception as e:
            self.error.emit(str(e))

    def rerank_with_hybrid_score(self, results: list, keywords: str) -> list:
        if not keywords: return results
        
        ocr_words = set(re.findall(r'\b\w+\b', keywords.lower()))
        ocr_words = {w for w in ocr_words if len(w) > 2}
        
        for res in results:
            title_words = set(re.findall(r'\b\w+\b', res['title'].lower()))
            
            if not title_words:
                text_score = 0.0
            else:
                matches = len(ocr_words.intersection(title_words))
                text_score = matches / len(ocr_words) if ocr_words else 0.0
            
            boosted_score = res['similarity_score'] + (text_score * 0.4)
            res['similarity_score'] = min(1.0, boosted_score)
            
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results

# --- Main Application Window ---
class LodeMetricsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LodeMetrics | Autonomous Resale Comps")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0;")

        self.db = LodeDB()
        self.vision_engine = VisionEngine(self.db)
        self.metadata_extractor = MetadataExtractor() 
        self.current_image_path = None
        self.current_keyword = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- TOP SECTION: Drop Zone & Button ---
        top_section = QHBoxLayout()
        
        left_panel = QVBoxLayout()
        
        # 1. Drop Zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.handle_image_drop)
        left_panel.addWidget(self.drop_zone)

        # 2. Analyze Button
        self.analyze_btn = QPushButton("🔍 Find Visual Comps")
        self.analyze_btn.setStyleSheet("""
            QPushButton { background-color: #0078d7; color: white; border: none; padding: 12px; font-size: 16px; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.analyze_btn.clicked.connect(self.start_ocr_flow)
        self.analyze_btn.setEnabled(False)
        left_panel.addWidget(self.analyze_btn)

        top_section.addLayout(left_panel, 1)

        # 3. Progress Section
        progress_section = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 5px;
            }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        progress_section.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaa; font-size: 14px; padding: 5px;")
        progress_section.addWidget(self.status_label)
        
        progress_section.addStretch()
        top_section.addLayout(progress_section, 1)

        main_layout.addLayout(top_section)

        # --- MIDDLE SECTION: Metadata & Optimizer ---
        mid_section = QHBoxLayout()
        
        # Metadata
        meta_group = QGroupBox("📋 Card Metadata")
        meta_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; }")
        meta_layout = QFormLayout()
        
        self.meta_source_lbl = QLabel("Awaiting Analysis...")
        meta_layout.addRow("Source:", self.meta_source_lbl)
        
        self.meta_serial_lbl = QLabel("-")
        meta_layout.addRow("Serial:", self.meta_serial_lbl)
        
        self.meta_auto_lbl = QLabel("-")
        meta_layout.addRow("Auto:", self.meta_auto_lbl)
        
        self.meta_rookie_lbl = QLabel("-")
        meta_layout.addRow("Rookie:", self.meta_rookie_lbl)
        
        meta_group.setLayout(meta_layout)
        mid_section.addWidget(meta_group, 1)
        
        # Optimizer
        opt_group = QGroupBox("📝 Keyword Optimizer")
        opt_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; }")
        opt_layout = QVBoxLayout()
        
        self.optimizer_text = QTextEdit()
        self.optimizer_text.setReadOnly(True)
        self.optimizer_text.setStyleSheet("background-color: #2b2b2b; border: none; color: #0f0; font-family: Consolas;")
        opt_layout.addWidget(self.optimizer_text)
        
        opt_group.setLayout(opt_layout)
        mid_section.addWidget(opt_group, 2)
        
        main_layout.addLayout(mid_section)

        # --- BOTTOM SECTION: Results Table & Charts ---
        bottom_section = QHBoxLayout()
        
        # Charts
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(6, 3))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax1.set_facecolor('#2b2b2b')
        self.ax2.set_facecolor('#2b2b2b')
        self.ax1.tick_params(colors='white')
        self.ax2.tick_params(colors='white')
        self.ax1.set_title("Comp Prices", color='white')
        self.ax2.set_title("Days on Market", color='white')
        
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #2b2b2b; border-radius: 5px;")
        chart_layout.addWidget(self.canvas)
        
        bottom_section.addWidget(chart_widget, 1)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Score", "Price", "Title", "Platform", "Link"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #2b2b2b; border: 1px solid #444; gridline-color: #444; }
            QHeaderView::section { background-color: #333; color: white; padding: 5px; border: 1px solid #444; }
            QTableWidget::item { padding: 5px; color: #e0e0e0; }
        """)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.open_link)
        
        bottom_section.addWidget(self.table, 2)
        
        main_layout.addLayout(bottom_section)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Drop an image to begin.")

    def open_link(self, row, col):
        if col == 4: 
            item = self.table.item(row, col)
            url = item.data(Qt.UserRole) if item else None
            if url:
                webbrowser.open(url)

    def handle_image_drop(self, file_path: str):
        self.current_image_path = file_path
        self.drop_zone.set_image(file_path)
        self.analyze_btn.setEnabled(True)
        self.clear_all()
        self.status_bar.showMessage(f"🖼️ Image loaded: {os.path.basename(file_path)}")

    def start_ocr_flow(self):
        if not self.current_image_path:
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.status_label.setText("Running OCR...")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳ Processing...")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        
        # Run OCR
        initial_keywords = self.metadata_extractor.extract_card_keywords(self.current_image_path)
        
        self.progress_bar.setValue(30)
        self.status_label.setText("Review keywords...")
        
        # Show Edit Dialog
        dialog = OCREditDialog(initial_keywords, self)
        result = dialog.exec()
        
        if result == 1:  # Accepted
            final_keywords = dialog.get_keywords()
            if final_keywords:
                self.current_keyword = final_keywords
                self.start_autonomous_search(final_keywords)
            else:
                self.reset_ui()
        else:
            self.reset_ui()

    def start_autonomous_search(self, keywords: str):
        self.progress_bar.setValue(40)
        self.status_label.setText("Searching local database...")
        
        self.worker = AnalysisWorker(self.vision_engine, self.metadata_extractor, self.current_image_path, keywords, top_k=8)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_worker_progress(self, message: str, percentage: int):
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.status_bar.showMessage(message)

    def on_analysis_finished(self, results: list, metadata: dict, keyword_used: str):
        QApplication.restoreOverrideCursor()
        
        if metadata.get("needs_scrape"):
            # TRIGGER LIVE SCRAPE
            self.progress_bar.setValue(50)
            self.status_label.setText(f"Scraping live data for '{keyword_used}'...")
            
            self.scrape_worker = PipelineWorker(self.db, self.vision_engine, keyword_used, max_pages=2)
            self.scrape_worker.progress.connect(self.on_scrape_progress)
            self.scrape_worker.finished.connect(lambda: self.on_scrape_finished(keyword_used))
            self.scrape_worker.error.connect(self.on_worker_error)
            self.scrape_worker.start()
            return

        # We have results!
        self.progress_bar.setValue(100)
        self.status_label.setText("✅ Complete!")
        
        self.populate_results(results, metadata)
        self.reset_ui_buttons()

    def on_scrape_progress(self, message: str, percentage: int):
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.status_bar.showMessage(message)

    def on_scrape_finished(self, keyword: str):
        self.progress_bar.setValue(95)
        self.status_label.setText("Re-running search with new data...")
        
        # Small delay to ensure DB is ready
        QTimer.singleShot(500, lambda: self.start_autonomous_search(keyword))

    def on_worker_error(self, error_msg: str):
        QApplication.restoreOverrideCursor()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ Error: {error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")
        self.reset_ui_buttons()

    def populate_results(self, results: list, metadata: dict):
        if not results:
            self.meta_source_lbl.setText("No comps found")
            self.optimizer_text.setText("No results found. Try adjusting the OCR keywords.")
            return
        
        self.populate_table(results)
        self.update_charts(results)
        self.update_metadata(metadata)
        self.generate_optimizer_text(results)

    def update_metadata(self, metadata: dict):
        if not metadata or metadata.get("needs_scrape"):
            self.clear_metadata()
            return
            
        self.meta_source_lbl.setText(metadata.get("source", "Unknown"))
        self.meta_source_lbl.setStyleSheet("color: #0078d7; font-weight: bold;")
        
        self.meta_serial_lbl.setText(metadata.get("serial") or "-")
        self.meta_auto_lbl.setText("✅ Yes" if metadata.get("auto") else "❌ No")
        self.meta_auto_lbl.setStyleSheet("color: #0f0;" if metadata.get("auto") else "color: #f00;")
        self.meta_rookie_lbl.setText("✅ Yes" if metadata.get("rookie") else "❌ No")

    def clear_metadata(self):
        self.meta_source_lbl.setText("-")
        self.meta_source_lbl.setStyleSheet("color: #aaa;")
        self.meta_serial_lbl.setText("-")
        self.meta_auto_lbl.setText("-")
        self.meta_auto_lbl.setStyleSheet("color: #e0e0e0;")
        self.meta_rookie_lbl.setText("-")

    def populate_table(self, results: list):
        self.table.setRowCount(len(results))
        for row, res in enumerate(results):
            score_item = QTableWidgetItem(f"{res['similarity_score']:.2f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, score_item)
            
            price_item = QTableWidgetItem(f"${res['sold_price']:,.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, price_item)
            
            self.table.setItem(row, 2, QTableWidgetItem(res['title'][:60] + "..." if len(res['title']) > 60 else res['title']))
            self.table.setItem(row, 3, QTableWidgetItem("Mercari"))
            
            link_item = QTableWidgetItem("Open")
            link_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            link_item.setForeground(Qt.GlobalColor.cyan)
            link_item.setData(Qt.UserRole, res['url']) 
            self.table.setItem(row, 4, link_item)

    def update_charts(self, results: list):
        self.ax1.clear()
        self.ax2.clear()
        
        prices = [r['sold_price'] for r in results if r['sold_price'] > 0]
        
        if prices:
            self.ax1.hist(prices, bins=5, color='#0078d7', edgecolor='white')
            self.ax1.set_xlabel("Sold Price ($)", color='white')
            self.ax1.set_ylabel("Frequency", color='white')
        else:
            self.ax1.text(0.5, 0.5, "No Price Data", color='white', ha='center')

        self.ax2.text(0.5, 0.5, "DOM Pending", color='white', ha='center')
        
        self.fig.tight_layout()
        self.canvas.draw()

    def generate_optimizer_text(self, results: list):
        all_text = " ".join([r['title'] for r in results]).lower()
        words = re.findall(r'\b\w+\b', all_text)
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'size', 'new', 'like', 'used', 'item'} 
        
        word_counts = Counter([w for w in words if w not in stop_words and len(w) > 3])
        top_keywords = word_counts.most_common(10)
        
        output = "🎯 HIGH-CONVERTING KEYWORDS:\n\n"
        for word, count in top_keywords:
            bar = "█" * (count // 2)
            output += f"{word.upper():<15} [{count:02d}] {bar}\n"
            
        self.optimizer_text.setText(output)

    def clear_all(self):
        self.clear_metadata()
        self.table.setRowCount(0)
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.text(0.5, 0.5, "Awaiting Analysis...", color='gray', ha='center')
        self.ax2.text(0.5, 0.5, "Awaiting Analysis...", color='gray', ha='center')
        self.canvas.draw()
        self.optimizer_text.clear()

    def reset_ui(self):
        QApplication.restoreOverrideCursor()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        self.reset_ui_buttons()

    def reset_ui_buttons(self):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🔍 Find Visual Comps")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = LodeMetricsGUI()
    window.show()
    sys.exit(app.exec())