#!/usr/bin/env python3
"""
Enhanced GUI for Testing SiloFlow HTTP Service & Automated Data Retrieval
=======================================================================

This script provides a comprehensive graphical interface to:
1. Test HTTP service endpoints with file uploads
2. Run automated data retrieval from database
3. Explore database structure (granaries, silos, date ranges)
4. Monitor operations with real-time progress

Usage:
    python testingservice.py
"""

import sys
from pathlib import Path

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(service_dir))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import requests
import json
import os
import threading
from datetime import datetime
import subprocess
import re # Added for regex in _parse_granaries_output

class SiloFlowTester:
    def __init__(self, root):
        self.root = root
        self.root.title("SiloFlow - Complete Testing & Data Management Interface")
        self.root.geometry("1200x800")
        
        # Service configuration
        self.service_url = "http://localhost:8000"
        self.remote_service_url = "http://192.168.28.242:8000"  # Default remote URL
        
        # Initialize data storage
        self.retrieval_granaries_data = []  # List of (name, id) tuples for retrieval dropdown
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_http_service_tab()
        self.create_remote_client_tab()
        self.create_data_retrieval_tab()
        self.create_database_explorer_tab()
        self.create_logs_tab()
        
    def create_http_service_tab(self):
        """Create HTTP Service Testing tab"""
        http_frame = ttk.Frame(self.notebook)
        self.notebook.add(http_frame, text="🌐 HTTP Service Testing")
        
        # Configure grid
        http_frame.columnconfigure(1, weight=1)
        http_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(http_frame, text="HTTP Service Testing (Local & Remote)", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Service URL configuration
        ttk.Label(http_frame, text="Service URL:").grid(row=1, column=0, sticky="w", pady=5, padx=5)
        self.url_var = tk.StringVar(value=self.service_url)
        url_entry = ttk.Entry(http_frame, textvariable=self.url_var, width=50)
        url_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        
        # Quick URL buttons
        url_buttons_frame = ttk.Frame(http_frame)
        url_buttons_frame.grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(url_buttons_frame, text="Local", command=lambda: self.url_var.set("http://localhost:8000")).pack(side=tk.LEFT, padx=2)
        ttk.Button(url_buttons_frame, text="Remote", command=lambda: self.url_var.set(self.remote_service_url)).pack(side=tk.LEFT, padx=2)
        ttk.Button(url_buttons_frame, text="Test", command=self.test_connection).pack(side=tk.LEFT, padx=2)
        
        # File selection
        ttk.Label(http_frame, text="Data File:").grid(row=2, column=0, sticky="w", pady=5, padx=5)
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(http_frame, textvariable=self.file_var, width=50)
        file_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        ttk.Button(http_frame, text="Browse", command=self.browse_file).grid(row=2, column=2, padx=5, pady=5)
        
        # Endpoint selection
        ttk.Label(http_frame, text="Endpoint:").grid(row=3, column=0, sticky="w", pady=5, padx=5)
        self.endpoint_var = tk.StringVar(value="/pipeline")
        endpoint_combo = ttk.Combobox(
            http_frame,
            textvariable=self.endpoint_var,
            values=["/pipeline", "/process", "/train", "/forecast", "/models", "/health"],
            state="readonly",
            width=20
        )
        endpoint_combo.grid(row=3, column=1, sticky="w", pady=5, padx=5)
        
        # Send button
        ttk.Button(http_frame, text="Send Request", command=self.send_request, style="Accent.TButton").grid(row=3, column=2, padx=5, pady=5)
        
        # Response area
        response_frame = ttk.LabelFrame(http_frame, text="Response", padding="10")
        response_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=10, padx=5)
        response_frame.columnconfigure(0, weight=1)
        response_frame.rowconfigure(0, weight=1)
        
        self.http_response_text = scrolledtext.ScrolledText(response_frame, height=15, width=80)
        self.http_response_text.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.http_status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(http_frame, textvariable=self.http_status_var, relief=tk.SUNKEN)
        status_label.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
    def create_remote_client_tab(self):
        """Create Remote Client Testing tab"""
        remote_frame = ttk.Frame(self.notebook)
        self.notebook.add(remote_frame, text="🌍 Remote Client Testing")
        
        # Configure grid
        remote_frame.columnconfigure(1, weight=1)
        remote_frame.rowconfigure(5, weight=1)
        
        # Title
        title_label = ttk.Label(remote_frame, text="Remote Client Testing Interface", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Remote service configuration
        config_frame = ttk.LabelFrame(remote_frame, text="Remote Service Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="Remote Service URL:").grid(row=0, column=0, sticky="w", pady=2)
        self.remote_url_var = tk.StringVar(value=self.remote_service_url)
        remote_url_entry = ttk.Entry(config_frame, textvariable=self.remote_url_var, width=50)
        remote_url_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(config_frame, text="Test Connection", command=self.test_remote_connection).grid(row=0, column=2, padx=5, pady=2)
        
        # File selection for remote testing
        file_frame = ttk.LabelFrame(remote_frame, text="Data File Selection", padding="10")
        file_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Data File:").grid(row=0, column=0, sticky="w", pady=2)
        self.remote_file_var = tk.StringVar()
        remote_file_entry = ttk.Entry(file_frame, textvariable=self.remote_file_var, width=50)
        remote_file_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(file_frame, text="Browse", command=self.browse_remote_file).grid(row=0, column=2, padx=5, pady=2)
        
        # Endpoint selection for remote testing
        endpoint_frame = ttk.LabelFrame(remote_frame, text="Endpoint Testing", padding="10")
        endpoint_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        endpoint_frame.columnconfigure(1, weight=1)
        
        ttk.Label(endpoint_frame, text="Endpoint:").grid(row=0, column=0, sticky="w", pady=2)
        self.remote_endpoint_var = tk.StringVar(value="/pipeline")
        remote_endpoint_combo = ttk.Combobox(
            endpoint_frame,
            textvariable=self.remote_endpoint_var,
            values=["/pipeline", "/process", "/train", "/forecast", "/models", "/health"],
            state="readonly",
            width=20
        )
        remote_endpoint_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Quick test buttons
        quick_test_frame = ttk.Frame(endpoint_frame)
        quick_test_frame.grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(quick_test_frame, text="Test /health", command=lambda: self.test_remote_endpoint("/health")).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_test_frame, text="Test /models", command=lambda: self.test_remote_endpoint("/models")).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_test_frame, text="Send File", command=self.send_remote_request).pack(side=tk.LEFT, padx=2)
        
        # Batch testing section
        batch_frame = ttk.LabelFrame(remote_frame, text="Batch Testing", padding="10")
        batch_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        batch_frame.columnconfigure(1, weight=1)
        
        ttk.Label(batch_frame, text="Test all endpoints:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Button(batch_frame, text="Run Full Test Suite", command=self.run_remote_test_suite, style="Accent.TButton").grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(batch_frame, text="Generate Test Report", command=self.generate_remote_test_report).grid(row=0, column=2, padx=5, pady=2)
        
        # Response area
        response_frame = ttk.LabelFrame(remote_frame, text="Remote Test Results", padding="10")
        response_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=5, padx=5)
        response_frame.columnconfigure(0, weight=1)
        response_frame.rowconfigure(0, weight=1)
        
        self.remote_response_text = scrolledtext.ScrolledText(response_frame, height=15, width=80)
        self.remote_response_text.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.remote_status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(remote_frame, textvariable=self.remote_status_var, relief=tk.SUNKEN)
        status_label.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
    def create_data_retrieval_tab(self):
        """Create Automated Data Retrieval tab"""
        retrieval_frame = ttk.Frame(self.notebook)
        self.notebook.add(retrieval_frame, text="Data Retrieval")
        
        # Configure grid
        retrieval_frame.columnconfigure(1, weight=1)
        retrieval_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(retrieval_frame, text="Automated Data Retrieval System", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Configuration section
        config_frame = ttk.LabelFrame(retrieval_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="Config JSON:").grid(row=0, column=0, sticky="w", pady=2)
        # Set default config path to the correct location
        default_config = str(Path(__file__).parent.parent.parent / "config" / "streaming_config.json")
        self.retrieval_cfg_var = tk.StringVar(value=default_config)
        cfg_entry = ttk.Entry(config_frame, textvariable=self.retrieval_cfg_var, width=50)
        cfg_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self._browse_retrieval_cfg).grid(row=0, column=2, padx=5, pady=2)
        
        # Retrieval options section
        options_frame = ttk.LabelFrame(retrieval_frame, text="Retrieval Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        options_frame.columnconfigure(1, weight=1)
        
        # Retrieval mode
        ttk.Label(options_frame, text="Retrieval Mode:").grid(row=0, column=0, sticky="w", pady=2)
        self.retrieval_mode_var = tk.StringVar(value="incremental")
        mode_combo = ttk.Combobox(
            options_frame,
            textvariable=self.retrieval_mode_var,
            values=["incremental", "full-retrieval", "date-range"],
            state="readonly",
            width=15
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        mode_combo.bind('<<ComboboxSelected>>', self._on_mode_change)
        
        # Granary selection
        ttk.Label(options_frame, text="Granary Selection:").grid(row=1, column=0, sticky="w", pady=2)
        self.retrieval_granary_var = tk.StringVar()
        self.retrieval_granary_combo = ttk.Combobox(options_frame, textvariable=self.retrieval_granary_var, width=30, state="readonly")
        self.retrieval_granary_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(options_frame, text="📋 Load Granaries", command=self.load_granaries_for_retrieval).grid(row=1, column=2, padx=5, pady=2)
        
        # Days for incremental
        ttk.Label(options_frame, text="Days (incremental):").grid(row=2, column=0, sticky="w", pady=2)
        self.days_var = tk.StringVar(value="7")
        days_entry = ttk.Entry(options_frame, textvariable=self.days_var, width=10)
        days_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Date range (initially hidden)
        self.date_range_frame = ttk.Frame(options_frame)
        self.date_range_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        self.date_range_frame.grid_remove()  # Hidden by default
        
        ttk.Label(self.date_range_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="w")
        self.start_date_var = tk.StringVar()
        ttk.Entry(self.date_range_frame, textvariable=self.start_date_var, width=15).grid(row=0, column=1, sticky="w", padx=2)
        
        ttk.Label(self.date_range_frame, text="End Date (YYYY-MM-DD):").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.end_date_var = tk.StringVar()
        ttk.Entry(self.date_range_frame, textvariable=self.end_date_var, width=15).grid(row=0, column=3, sticky="w", padx=2)
        
        # Options
        self.cleanup_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Cleanup old files after retrieval", variable=self.cleanup_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        
        # Run button
        ttk.Button(retrieval_frame, text="Run Automated Data Retrieval", command=self.run_automated_retrieval, style="Accent.TButton").grid(row=3, column=1, pady=10)
        
        # Response area
        response_frame = ttk.LabelFrame(retrieval_frame, text="Retrieval Output", padding="10")
        response_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=5, padx=5)
        response_frame.columnconfigure(0, weight=1)
        response_frame.rowconfigure(0, weight=1)
        
        self.retrieval_response_text = scrolledtext.ScrolledText(response_frame, height=15, width=80)
        self.retrieval_response_text.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.retrieval_status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(retrieval_frame, textvariable=self.retrieval_status_var, relief=tk.SUNKEN)
        status_label.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
    def create_database_explorer_tab(self):
        """Create Database Explorer tab"""
        explorer_frame = ttk.Frame(self.notebook)
        self.notebook.add(explorer_frame, text="Database Explorer")
        
        # Configure grid
        explorer_frame.columnconfigure(1, weight=1)
        explorer_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(explorer_frame, text="Database Structure Explorer", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Configuration section
        config_frame = ttk.LabelFrame(explorer_frame, text="Database Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="Config JSON:").grid(row=0, column=0, sticky="w", pady=2)
        # Set default config path to the correct location
        default_config = str(Path(__file__).parent.parent.parent / "config" / "streaming_config.json")
        self.explorer_cfg_var = tk.StringVar(value=default_config)
        cfg_entry = ttk.Entry(config_frame, textvariable=self.explorer_cfg_var, width=50)
        cfg_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self._browse_explorer_cfg).grid(row=0, column=2, padx=5, pady=2)
        
        # Step 1: Get All Granaries
        step1_frame = ttk.LabelFrame(explorer_frame, text="Step 1: Get All Granaries", padding="10")
        step1_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        step1_frame.columnconfigure(1, weight=1)
        
        ttk.Button(step1_frame, text="Get All Granaries", command=self.get_all_granaries, style="Accent.TButton").grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(step1_frame, text="🔗 Test Database Connection", command=self.test_db_connection).grid(row=0, column=1, padx=5, pady=5)
        
        # Step 2: Select Granary and Get Silos
        step2_frame = ttk.LabelFrame(explorer_frame, text="Step 2: Select Granary & Get Silos", padding="10")
        step2_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        step2_frame.columnconfigure(1, weight=1)
        
        ttk.Label(step2_frame, text="Granary:").grid(row=0, column=0, sticky="w", pady=2)
        self.granary_selection_var = tk.StringVar()
        self.granary_combo = ttk.Combobox(step2_frame, textvariable=self.granary_selection_var, width=40, state="readonly")
        self.granary_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(step2_frame, text="📦 Get Silos for Selected Granary", command=self.get_silos_for_granary, style="Accent.TButton").grid(row=0, column=2, padx=5, pady=2)
        
        # Step 3: Select Silo and Get Date Range
        step3_frame = ttk.LabelFrame(explorer_frame, text="Step 3: Select Silo & Get Date Range", padding="10")
        step3_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        step3_frame.columnconfigure(1, weight=1)
        
        ttk.Label(step3_frame, text="Silo:").grid(row=0, column=0, sticky="w", pady=2)
        self.silo_selection_var = tk.StringVar()
        self.silo_combo = ttk.Combobox(step3_frame, textvariable=self.silo_selection_var, width=40, state="readonly")
        self.silo_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(step3_frame, text="📅 Get Date Range for Selected Silo", command=self.get_date_range_for_silo, style="Accent.TButton").grid(row=0, column=2, padx=5, pady=2)
        
        # Legacy actions (for backward compatibility)
        legacy_frame = ttk.LabelFrame(explorer_frame, text="Legacy Actions", padding="10")
        legacy_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        legacy_frame.columnconfigure(1, weight=1)
        
        ttk.Button(legacy_frame, text="📋 List All Granaries & Silos", command=self.list_granaries_silos).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(legacy_frame, text="📅 Get All Date Ranges", command=self.get_date_ranges).grid(row=0, column=1, padx=5, pady=5)
        
        # Results area
        results_frame = ttk.LabelFrame(explorer_frame, text="Exploration Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=5, padx=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.explorer_response_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.explorer_response_text.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.explorer_status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(explorer_frame, textvariable=self.explorer_status_var, relief=tk.SUNKEN)
        status_label.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
        # Store data for selections
        self.granaries_data = []
        self.silos_data = []
        
    def create_logs_tab(self):
        """Create Logs and Monitoring tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs & Monitoring")
        
        # Configure grid
        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(logs_frame, text="System Logs & Monitoring", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Logs area
        logs_area = ttk.LabelFrame(logs_frame, text="System Logs", padding="10")
        logs_area.grid(row=1, column=0, sticky="nsew", pady=5, padx=5)
        logs_area.columnconfigure(0, weight=1)
        logs_area.rowconfigure(0, weight=1)
        
        self.logs_text = scrolledtext.ScrolledText(logs_area, height=20, width=100)
        self.logs_text.grid(row=0, column=0, sticky="nsew")
        
        # Add initial content
        self.logs_text.insert(tk.END, "SiloFlow System Logs\n")
        self.logs_text.insert(tk.END, "=" * 50 + "\n\n")
        self.logs_text.insert(tk.END, "📋 System Information:\n")
        self.logs_text.insert(tk.END, f"• Application: SiloFlow Testing Interface\n")
        self.logs_text.insert(tk.END, f"• Version: 2.0.0\n")
        self.logs_text.insert(tk.END, f"• Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logs_text.insert(tk.END, f"• Python: {sys.version}\n\n")
        self.logs_text.insert(tk.END, "📁 Directory Structure:\n")
        self.logs_text.insert(tk.END, f"• Current Directory: {Path.cwd()}\n")
        self.logs_text.insert(tk.END, f"• Service Directory: {Path(__file__).parent}\n")
        self.logs_text.insert(tk.END, f"• Models Directory: {Path('models').absolute()}\n")
        self.logs_text.insert(tk.END, f"• Data Directory: {Path('data').absolute()}\n\n")
        self.logs_text.insert(tk.END, "Ready for operations...\n\n")
        
    def test_connection(self):
        """Test HTTP service connection"""
        try:
            url = self.url_var.get().rstrip('/')
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                messagebox.showinfo("Success", "HTTP service is running and healthy!")
                self.http_status_var.set("Service connected successfully")
            else:
                messagebox.showerror("Error", f"Service returned status code: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Connection Error", f"Could not connect to service: {str(e)}")
            
    def browse_file(self):
        """Open file dialog to select data file (CSV or Parquet)"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("Data files", "*.csv;*.parquet"), 
                ("CSV files", "*.csv"), 
                ("Parquet files", "*.parquet"), 
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_var.set(filename)
            # Show file info
            if filename.lower().endswith('.parquet'):
                self._show_parquet_info(filename, self.http_response_text)
            elif filename.lower().endswith('.csv'):
                self._show_csv_info(filename, self.http_response_text)
    
    def _show_parquet_info(self, filepath, text_widget):
        """Show information about selected Parquet file"""
        try:
            import pandas as pd
            df = pd.read_parquet(filepath)
            
            # Update status with file info
            file_size = os.path.getsize(filepath)
            info_text = f"Parquet file loaded: {len(df):,} rows, {len(df.columns)} columns, {file_size:,} bytes"
            self.http_status_var.set(info_text)
            
            # Show file info in response area
            text_widget.insert(tk.END, f"Parquet File Info:\n")
            text_widget.insert(tk.END, f"   File: {os.path.basename(filepath)}\n")
            text_widget.insert(tk.END, f"   Rows: {len(df):,}\n")
            text_widget.insert(tk.END, f"   Columns: {len(df.columns)}\n")
            text_widget.insert(tk.END, f"   Size: {file_size:,} bytes\n")
            text_widget.insert(tk.END, f"   Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}\n\n")
            
        except Exception as e:
            text_widget.insert(tk.END, f"Could not read Parquet file info: {str(e)}\n\n")
    
    def _show_csv_info(self, filepath, text_widget):
        """Show information about selected CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            
            # Update status with file info
            file_size = os.path.getsize(filepath)
            info_text = f"CSV file loaded: {len(df):,} rows, {len(df.columns)} columns, {file_size:,} bytes"
            self.http_status_var.set(info_text)
            
            # Show file info in response area
            text_widget.insert(tk.END, f"CSV File Info:\n")
            text_widget.insert(tk.END, f"   File: {os.path.basename(filepath)}\n")
            text_widget.insert(tk.END, f"   Rows: {len(df):,}\n")
            text_widget.insert(tk.END, f"   Columns: {len(df.columns)}\n")
            text_widget.insert(tk.END, f"   Size: {file_size:,} bytes\n")
            text_widget.insert(tk.END, f"   Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}\n\n")
            
        except Exception as e:
            text_widget.insert(tk.END, f"Could not read CSV file info: {str(e)}\n\n")
            
    def send_request(self):
        """Send the selected file to the HTTP service"""
        # Get configuration
        service_url = self.url_var.get().rstrip('/')
        endpoint = self.endpoint_var.get()
        file_path = self.file_var.get()

        # Determine if the chosen endpoint needs a file upload
        file_required = endpoint in ["/pipeline"]

        if file_required and not file_path:
            messagebox.showerror("Error", "Please select a data file (CSV or Parquet)")
            return

        # Clear response area
        self.http_response_text.delete(1.0, tk.END)
        self.http_response_text.insert(tk.END, f"Sending request to {service_url}{endpoint}\n")
        self.http_response_text.insert(tk.END, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.http_response_text.insert(tk.END, "-" * 60 + "\n\n")

        # Launch request in background thread
        thread = threading.Thread(target=self._send_request_thread, args=(service_url, endpoint, file_path))
        thread.daemon = True
        thread.start()

    def _send_request_thread(self, service_url, endpoint, file_path):
        """Send request in background thread"""
        try:
            full_url = f"{service_url}{endpoint}"
            timeout_seconds = 10800  # 3 hours for long operations

            if file_path:
                with open(file_path, "rb") as f:
                    file_ext = os.path.splitext(file_path)[1].lower()
                    content_type = "application/octet-stream" if file_ext == ".parquet" else "text/csv"
                    files = {"file": (os.path.basename(file_path), f, content_type)}
                    response = requests.post(full_url, files=files, timeout=timeout_seconds)
            else:
                response = requests.get(full_url, timeout=30)
            
            # Update GUI with response
            self.root.after(0, self._update_http_response, response)
            
        except requests.exceptions.ConnectionError:
            self.root.after(0, self._show_error, "Connection Error", 
                          "Could not connect to the service. Make sure it's running on the specified URL.")
        except requests.exceptions.Timeout:
            self.root.after(0, self._show_error, "Timeout Error", 
                          "Request timed out after 3 hours. The service may still be processing.")
        except Exception as e:
            self.root.after(0, self._show_error, "Error", f"An error occurred: {str(e)}")
            
    def _update_http_response(self, response):
        """Update GUI with HTTP response results"""
        try:
            # Display response details
            self.http_response_text.insert(tk.END, f"Status Code: {response.status_code}\n")
            self.http_response_text.insert(tk.END, f"Response Time: {response.elapsed.total_seconds():.2f}s\n")
            self.http_response_text.insert(tk.END, f"Content-Type: {response.headers.get('content-type', 'Unknown')}\n")
            self.http_response_text.insert(tk.END, "-" * 50 + "\n\n")
            
            content_type = response.headers.get('content-type', '').lower()
            if 'application/json' in content_type:
                # Pretty print JSON
                try:
                    json_response = response.json()
                    import json
                    formatted_json = json.dumps(json_response, indent=2, ensure_ascii=False)
                    self.http_response_text.insert(tk.END, formatted_json)
                except Exception as e:
                    self.http_response_text.insert(tk.END, f"Error parsing JSON: {str(e)}\n")
            elif 'application/octet-stream' in content_type or 'parquet' in content_type:
                # Parquet file: show summary only
                self.http_response_text.insert(tk.END, f"Received Parquet file ({len(response.content):,} bytes).\n")
                # Try to decode summary from header
                summary_b64 = response.headers.get('x-forecast-summary')
                if summary_b64:
                    import base64, json
                    try:
                        summary_json = base64.b64decode(summary_b64).decode('utf-8')
                        summary = json.loads(summary_json)
                        self.http_response_text.insert(tk.END, "Summary (from header):\n")
                        self.http_response_text.insert(tk.END, json.dumps(summary, indent=2, ensure_ascii=False))
                        self.http_response_text.insert(tk.END, "\n")
                    except Exception as e:
                        self.http_response_text.insert(tk.END, f"Could not decode summary: {str(e)}\n")
                self.http_response_text.insert(tk.END, "(Parquet file not displayed. Save from API if needed.)\n")
            elif 'text/csv' in content_type:
                self._display_csv_response(response, self.http_response_text)
            else:
                # Fallback: display as text
                self.http_response_text.insert(tk.END, response.text)
                
            # Update status
            if response.status_code == 200:
                self.http_status_var.set(f"Success - {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.http_status_var.set(f"Error {response.status_code} - {datetime.now().strftime('%H:%M:%S')}")
                
        except Exception as e:
            self.http_response_text.insert(tk.END, f"Error processing response: {str(e)}")
            self.http_status_var.set("Error processing response")
    
    def _display_csv_response(self, response, text_widget):
        """Display CSV response content directly in the GUI"""
        try:
            # Decode CSV content
            csv_content = response.content.decode('utf-8')
            
            # Display file info
            text_widget.insert(tk.END, f"CSV Response Data:\n")
            text_widget.insert(tk.END, f"File size: {len(response.content):,} bytes\n")
            text_widget.insert(tk.END, "-" * 50 + "\n")
            
            # Display all CSV content
            text_widget.insert(tk.END, csv_content)
            
            text_widget.insert(tk.END, "\n" + "-" * 50 + "\n")
            
            # Show summary information
            lines = csv_content.split('\n')
            if lines and lines[0].strip():
                headers = lines[0].split(',')
                data_rows = len([line for line in lines[1:] if line.strip()])
                text_widget.insert(tk.END, f"Columns: {len(headers)}\n")
                text_widget.insert(tk.END, f"Total data rows: {data_rows:,}\n\n")
            
        except Exception as e:
            text_widget.insert(tk.END, f"Error displaying CSV response: {str(e)}\n\n")
            
    def _show_error(self, title, message):
        """Show error message"""
        messagebox.showerror(title, message)
        self.http_status_var.set("Error")
        self.http_response_text.insert(tk.END, f"Error: {message}\n")

    # ------------------------------------------------------------------
    # AUTOMATED DATA RETRIEVAL HELPERS
    # ------------------------------------------------------------------

    def _on_mode_change(self, event=None):
        """Show/hide date range fields based on selected mode"""
        mode = self.retrieval_mode_var.get()
        if mode == "date-range":
            self.date_range_frame.grid()
        else:
            self.date_range_frame.grid_remove()
        
    def _browse_retrieval_cfg(self):
        filename = filedialog.askopenfilename(
            title="Select Config JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.retrieval_cfg_var.set(filename)

    def load_granaries_for_retrieval(self):
        """Load granaries and populate the retrieval dropdown"""
        config_file = self.retrieval_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
        
        # Clear response area
        self.retrieval_response_text.delete(1.0, tk.END)
        self.retrieval_response_text.insert(tk.END, "📋 Loading Granaries for Data Retrieval...\n")
        self.retrieval_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.retrieval_response_text.insert(tk.END, "-" * 60 + "\n\n")
        
        # Build command using list_granaries.py
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "database" / "list_granaries.py"
        cmd = ["python", str(script_path), "--config", config_file]
        
        self.retrieval_response_text.insert(tk.END, f"Running command: {' '.join(cmd)}\n\n")
        self.retrieval_status_var.set("Loading granaries...")
        
        thread = threading.Thread(target=self._run_retrieval_granaries_subprocess, args=(cmd,))
        thread.daemon = True
        thread.start()

    def _run_retrieval_granaries_subprocess(self, cmd):
        """Execute the get granaries command and populate retrieval dropdown."""
        try:
            from utils.database_utils import SubprocessUtils
            
            success, error_msg, output_lines = SubprocessUtils.run_subprocess(cmd, "get granaries for retrieval")
            
            # Update GUI with output in real-time
            for line in output_lines:
                self.root.after(0, self.retrieval_response_text.insert, tk.END, line)
                self.root.after(0, self.retrieval_response_text.see, tk.END)
            
            self.root.after(0, self.retrieval_status_var.set, "")
            
            if success:
                # Parse output to extract granary names and populate dropdown
                self.root.after(0, self._parse_retrieval_granaries_output, output_lines)
                self.root.after(0, self.retrieval_status_var.set, "Granaries loaded successfully")
                self.root.after(0, self.retrieval_response_text.insert, tk.END, "\nGranaries loaded successfully!\n")
            else:
                self.root.after(0, self.retrieval_status_var.set, f"Operation failed: {error_msg}")
                self.root.after(0, self.retrieval_response_text.insert, tk.END, f"\nOperation failed: {error_msg}\n")
                
        except Exception as exc:
            self.root.after(0, self._show_retrieval_error, "Retrieval Error", str(exc))

    def _parse_retrieval_granaries_output(self, output_lines):
        """Parse the granaries output and populate the retrieval dropdown."""
        try:
            granary_data = []  # List of (name, id) tuples
            for line in output_lines:
                # The list_granaries.py script outputs in a space-padded format
                # Format: "ID                                    Name                     Table   Silos"
                # Example: "some-long-id                          中正粮食储备库              77      8"
                
                # Skip header lines and separator lines
                if (line.startswith('ID') or line.startswith('-') or 
                    line.startswith('Available') or line.startswith('=') or
                    'granary_id' in line.lower() or 'granary_name' in line.lower()):
                    continue
                
                # Look for lines that contain granary data (should have at least 3 parts when split)
                if len(line.strip()) > 0 and not line.startswith('#'):
                    # Split by multiple spaces and filter out empty parts
                    parts = [part.strip() for part in line.split() if part.strip()]
                    if len(parts) >= 3:
                        # The first part is the ID, second part is typically the name
                        granary_id = parts[0]
                        # Look for the granary name (Chinese characters or longer text)
                        granary_name = None
                        for part in parts[1:]:  # Skip the ID
                            if (len(part) > 2 and 
                                (any('\u4e00' <= char <= '\u9fff' for char in part) or  # Chinese characters
                                 len(part) > 10)):  # Long names
                                granary_name = part
                                break
                        
                        if granary_name and granary_name != 'Name':
                            granary_data.append((granary_name, granary_id))
                
                # Also look for lines that might contain granary names in other formats
                elif 'Granary:' in line:
                    # Extract granary name from lines like "Granary: GranaryName (ID: xxx, Table: xxx)"
                    parts = line.split('Granary:')
                    if len(parts) > 1:
                        granary_part = parts[1].split('(ID:')[0].strip()
                        if granary_part:
                            # Try to extract ID from the same line
                            id_match = re.search(r'ID:\s*([^\s,]+)', line)
                            granary_id = id_match.group(1) if id_match else granary_part
                            granary_data.append((granary_part, granary_id))
            
            # Store granary data and populate dropdown
            self.retrieval_granaries_data = granary_data  # List of (name, id) tuples
            granary_names = [item[0] for item in granary_data]  # Just the names for display
            self.retrieval_granary_combo['values'] = granary_names
            if granary_names:
                self.retrieval_granary_combo.set(granary_names[0])  # Set first granary as default
            
            self.retrieval_response_text.insert(tk.END, f"\n📋 Found {len(granary_names)} granaries\n")
            if granary_names:
                self.retrieval_response_text.insert(tk.END, f"Granaries: {', '.join(granary_names[:5])}")
                if len(granary_names) > 5:
                    self.retrieval_response_text.insert(tk.END, f" and {len(granary_names) - 5} more...\n")
                else:
                    self.retrieval_response_text.insert(tk.END, "\n")
            else:
                self.retrieval_response_text.insert(tk.END, "No granaries found in output\n")
                self.retrieval_response_text.insert(tk.END, "\nDEBUG: Raw output for troubleshooting:\n")
                self.retrieval_response_text.insert(tk.END, "-" * 40 + "\n")
                for i, line in enumerate(output_lines[:10]):
                    self.retrieval_response_text.insert(tk.END, f"{i+1:2d}: {repr(line)}\n")
                if len(output_lines) > 10:
                    self.retrieval_response_text.insert(tk.END, f"... and {len(output_lines) - 10} more lines\n")
            
        except Exception as e:
            self.retrieval_response_text.insert(tk.END, f"\nError parsing granaries: {str(e)}\n")
            self.retrieval_response_text.insert(tk.END, f"Raw output lines: {output_lines[:5]}\n")

    def run_automated_retrieval(self):
        """Launch automated_data_retrieval.py in a background thread."""
        
        # Validate inputs
        mode = self.retrieval_mode_var.get()
        config_file = self.retrieval_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
            
        if mode == "date-range":
            start_date = self.start_date_var.get().strip()
            end_date = self.end_date_var.get().strip()
            if not start_date or not end_date:
                messagebox.showerror("Error", "Start and end dates are required for date-range mode")
                return
        
        # Build command
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "data_retrieval" / "automated_data_retrieval.py"
        cmd = ["python", str(script_path), "--config", config_file]
        
        # Add mode-specific arguments
        if mode == "incremental":
            days = self.days_var.get().strip()
            if not days or not days.isdigit():
                messagebox.showerror("Error", "Days must be a valid number")
                return
            cmd.extend(["--incremental", "--days", days])
        elif mode == "full-retrieval":
            cmd.append("--full-retrieval")
        elif mode == "date-range":
            cmd.extend(["--date-range", "--start", start_date, "--end", end_date])
        
        # Add granary filter if specified
        selected_granary_name = self.retrieval_granary_var.get().strip()
        if selected_granary_name:
            # Find the granary ID for the selected name
            selected_granary_id = None
            if hasattr(self, 'retrieval_granaries_data'):
                for name, granary_id in self.retrieval_granaries_data:
                    if name == selected_granary_name:
                        selected_granary_id = granary_id
                        break
            
            if selected_granary_id:
                cmd.extend(["--granary", selected_granary_id])
            else:
                # Fallback to using the name if ID not found
                cmd.extend(["--granary", selected_granary_name])
        
        # Add cleanup option
        if self.cleanup_var.get():
            cmd.extend(["--cleanup"])
        
        # Clear UI and launch thread
        self.retrieval_response_text.insert(tk.END, f"\nRunning Automated Data Retrieval:\n")
        self.retrieval_response_text.insert(tk.END, f"Command: {' '.join(cmd)}\n")
        self.retrieval_response_text.insert(tk.END, f"Mode: {mode}\n")
        if selected_granary_name:
            self.retrieval_response_text.insert(tk.END, f"Granary Filter: {selected_granary_name}\n")
        self.retrieval_response_text.insert(tk.END, "-" * 60 + "\n")
        
        self.retrieval_status_var.set("Starting Automated Data Retrieval...")

        thread = threading.Thread(target=self._run_retrieval_subprocess, args=(cmd,))
        thread.daemon = True
        thread.start()

    def _run_retrieval_subprocess(self, cmd):
        """Execute the automated data retrieval command and stream output."""
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            if process.stdout is None:
                self.root.after(0, self._show_retrieval_error, "Retrieval Error", "No output captured from subprocess")
                return
                
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Update GUI with the line
                    self.root.after(0, self.retrieval_response_text.insert, tk.END, line)
                    self.root.after(0, self.retrieval_response_text.see, tk.END)
                    
                    # Update progress for key messages
                    if "Starting" in line:
                        self.root.after(0, self.retrieval_status_var.set, line.strip())
                    elif "completed" in line.lower() or "finished" in line.lower():
                        self.root.after(0, self.retrieval_status_var.set, line.strip())
            
            process.wait()
            self.root.after(0, self.retrieval_status_var.set, "")
            
            if process.returncode == 0:
                self.root.after(0, self.retrieval_status_var.set, "Data retrieval completed successfully")
                self.root.after(0, self.retrieval_response_text.insert, tk.END, "\nData retrieval completed successfully!\n")
            else:
                self.root.after(0, self.retrieval_status_var.set, f"Data retrieval failed (exit code {process.returncode})")
                self.root.after(0, self.retrieval_response_text.insert, tk.END, f"\nData retrieval failed with exit code {process.returncode}\n")
                
        except Exception as exc:
            self.root.after(0, self._show_retrieval_error, "Retrieval Error", str(exc))

    def _show_retrieval_error(self, title, message):
        """Show error message for retrieval operations"""
        messagebox.showerror(title, message)
        self.retrieval_status_var.set("Error")

    # ------------------------------------------------------------------
    # DATABASE EXPLORER HELPERS
    # ------------------------------------------------------------------

    def _browse_explorer_cfg(self):
        filename = filedialog.askopenfilename(
            title="Select Config JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.explorer_cfg_var.set(filename)

    def list_granaries_silos(self):
        """List all granaries and their silos"""
        config_file = self.explorer_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
        
        # Clear response area
        self.explorer_response_text.delete(1.0, tk.END)
        self.explorer_response_text.insert(tk.END, "Listing Granaries and Silos...\n")
        self.explorer_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.explorer_response_text.insert(tk.END, "-" * 60 + "\n\n")
        
        # Build command
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "database" / "list_granaries.py"
        cmd = ["python", str(script_path), "--config", config_file]
        
        self.explorer_status_var.set("Listing granaries and silos...")
        
        thread = threading.Thread(target=self._run_explorer_subprocess, args=(cmd, "granaries"))
        thread.daemon = True
        thread.start()

    def get_date_ranges(self):
        """Get date ranges for all silos"""
        config_file = self.explorer_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
        
        # Clear response area
        self.explorer_response_text.delete(1.0, tk.END)
        self.explorer_response_text.insert(tk.END, "📅 Getting Date Ranges for All Silos...\n")
        self.explorer_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.explorer_response_text.insert(tk.END, "-" * 60 + "\n\n")
        
        # Build command using the new get_date_ranges.py script
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "database" / "get_date_ranges.py"
        cmd = ["python", str(script_path), "--config", config_file]
        
        self.explorer_status_var.set("Getting date ranges...")
        
        thread = threading.Thread(target=self._run_explorer_subprocess, args=(cmd, "dates"))
        thread.daemon = True
        thread.start()

    def test_db_connection(self):
        """Test database connection"""
        config_file = self.explorer_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
        
        # Clear response area
        self.explorer_response_text.delete(1.0, tk.END)
        self.explorer_response_text.insert(tk.END, "🔗 Testing Database Connection...\n")
        self.explorer_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.explorer_response_text.insert(tk.END, "-" * 60 + "\n\n")
        
        # Build command
        from pathlib import Path
        script_path = Path(__file__).resolve().parent / "test_connection.py"
        cmd = ["python", str(script_path)]
        
        self.explorer_status_var.set("Testing database connection...")
        
        thread = threading.Thread(target=self._run_explorer_subprocess, args=(cmd, "connection"))
        thread.daemon = True
        thread.start()

    def _run_explorer_subprocess(self, cmd, operation_type):
        """Execute database explorer commands and stream output."""
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            if process.stdout is None:
                self.root.after(0, self._show_explorer_error, "Explorer Error", "No output captured from subprocess")
                return
                
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Update GUI with the line
                    self.root.after(0, self.explorer_response_text.insert, tk.END, line)
                    self.root.after(0, self.explorer_response_text.see, tk.END)
                    
                    # Update progress for key messages
                    if "Starting" in line or "Testing" in line or "Listing" in line:
                        self.root.after(0, self.explorer_status_var.set, line.strip())
                    elif "completed" in line.lower() or "finished" in line.lower() or "successful" in line.lower():
                        self.root.after(0, self.explorer_status_var.set, line.strip())
            
            process.wait()
            self.root.after(0, self.explorer_status_var.set, "")
            
            if process.returncode == 0:
                if operation_type == "granaries":
                    self.root.after(0, self.explorer_status_var.set, "Granaries and silos listed successfully")
                    self.root.after(0, self.explorer_response_text.insert, tk.END, "\nGranaries and silos listed successfully!\n")
                elif operation_type == "dates":
                    self.root.after(0, self.explorer_status_var.set, "Date ranges retrieved successfully")
                    self.root.after(0, self.explorer_response_text.insert, tk.END, "\nDate ranges retrieved successfully!\n")
                elif operation_type == "connection":
                    self.root.after(0, self.explorer_status_var.set, "Database connection test completed")
                    self.root.after(0, self.explorer_response_text.insert, tk.END, "\nDatabase connection test completed!\n")
            else:
                self.root.after(0, self.explorer_status_var.set, f"Operation failed (exit code {process.returncode})")
                self.root.after(0, self.explorer_response_text.insert, tk.END, f"\nOperation failed with exit code {process.returncode}\n")
                
        except Exception as exc:
            self.root.after(0, self._show_explorer_error, "Explorer Error", str(exc))

    def _show_explorer_error(self, title, message):
        """Show error message for explorer operations"""
        messagebox.showerror(title, message)
        self.explorer_status_var.set("Error")

    def get_all_granaries(self):
        """Get all granaries and populate the selection dropdown"""
        config_file = self.explorer_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
        
        # Clear response area
        self.explorer_response_text.delete(1.0, tk.END)
        self.explorer_response_text.insert(tk.END, "Getting All Granaries...\n")
        self.explorer_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.explorer_response_text.insert(tk.END, "-" * 60 + "\n\n")
        
        # Build command using list_granaries.py
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "database" / "list_granaries.py"
        cmd = ["python", str(script_path), "--config", config_file]
        
        self.explorer_response_text.insert(tk.END, f"Running command: {' '.join(cmd)}\n\n")
        self.explorer_status_var.set("Getting all granaries...")
        
        thread = threading.Thread(target=self._run_get_granaries_subprocess, args=(cmd,))
        thread.daemon = True
        thread.start()

    def _run_get_granaries_subprocess(self, cmd):
        """Execute the get granaries command and populate dropdown."""
        try:
            from utils.database_utils import SubprocessUtils
            
            success, error_msg, output_lines = SubprocessUtils.run_subprocess(cmd, "get granaries")
            
            # Update GUI with output in real-time
            for line in output_lines:
                self.root.after(0, self.explorer_response_text.insert, tk.END, line)
                self.root.after(0, self.explorer_response_text.see, tk.END)
            
            self.root.after(0, self.explorer_status_var.set, "")
            
            if success:
                # Parse output to extract granary names and populate dropdown
                self.root.after(0, self._parse_granaries_output, output_lines)
                self.root.after(0, self.explorer_status_var.set, "Granaries loaded successfully")
                self.root.after(0, self.explorer_response_text.insert, tk.END, "\nGranaries loaded successfully!\n")
            else:
                self.root.after(0, self.explorer_status_var.set, f"Operation failed: {error_msg}")
                self.root.after(0, self.explorer_response_text.insert, tk.END, f"\n❌ Operation failed: {error_msg}\n")
                
        except Exception as exc:
            self.root.after(0, self._show_explorer_error, "Explorer Error", str(exc))

    def _parse_granaries_output(self, output_lines):
        """Parse the granaries output and populate the dropdown."""
        try:
            granary_data = []  # List of (name, id) tuples
            for line in output_lines:
                # The list_granaries.py script outputs in a space-padded format
                # Format: "ID                                    Name                     Table   Silos"
                # Example: "some-long-id                          中正粮食储备库              77      8"
                
                # Skip header lines and separator lines
                if (line.startswith('ID') or line.startswith('-') or 
                    line.startswith('Available') or line.startswith('=') or
                    'granary_id' in line.lower() or 'granary_name' in line.lower()):
                    continue
                
                # Look for lines that contain granary data (should have at least 3 parts when split)
                if len(line.strip()) > 0 and not line.startswith('#'):
                    # Split by multiple spaces and filter out empty parts
                    parts = [part.strip() for part in line.split() if part.strip()]
                    if len(parts) >= 3:
                        # The first part is the ID, second part is typically the name
                        granary_id = parts[0]
                        # Look for the granary name (Chinese characters or longer text)
                        granary_name = None
                        for part in parts[1:]:  # Skip the ID
                            if (len(part) > 2 and 
                                (any('\u4e00' <= char <= '\u9fff' for char in part) or  # Chinese characters
                                 len(part) > 10)):  # Long names
                                granary_name = part
                                break
                        
                        if granary_name and granary_name != 'Name':
                            granary_data.append((granary_name, granary_id))
                
                # Also look for lines that might contain granary names in other formats
                elif 'Granary:' in line:
                    # Extract granary name from lines like "Granary: GranaryName (ID: xxx, Table: xxx)"
                    parts = line.split('Granary:')
                    if len(parts) > 1:
                        granary_part = parts[1].split('(ID:')[0].strip()
                        if granary_part:
                            # Try to extract ID from the same line
                            id_match = re.search(r'ID:\s*([^\s,]+)', line)
                            granary_id = id_match.group(1) if id_match else granary_part
                            granary_data.append((granary_part, granary_id))
            
            # Store granary data and populate dropdown
            self.granaries_data = granary_data  # List of (name, id) tuples
            granary_names = [item[0] for item in granary_data]  # Just the names for display
            self.granary_combo['values'] = granary_names
            if granary_names:
                self.granary_combo.set(granary_names[0])  # Set first granary as default
            
            self.explorer_response_text.insert(tk.END, f"\n📋 Found {len(granary_names)} granaries\n")
            if granary_names:
                self.explorer_response_text.insert(tk.END, f"Granaries: {', '.join(granary_names[:5])}")
                if len(granary_names) > 5:
                    self.explorer_response_text.insert(tk.END, f" and {len(granary_names) - 5} more...\n")
                else:
                    self.explorer_response_text.insert(tk.END, "\n")
            else:
                self.explorer_response_text.insert(tk.END, "No granaries found in output\n")
                self.explorer_response_text.insert(tk.END, "\n🔍 DEBUG: Raw output for troubleshooting:\n")
                self.explorer_response_text.insert(tk.END, "-" * 40 + "\n")
                for i, line in enumerate(output_lines[:10]):
                    self.explorer_response_text.insert(tk.END, f"{i+1:2d}: {repr(line)}\n")
                if len(output_lines) > 10:
                    self.explorer_response_text.insert(tk.END, f"... and {len(output_lines) - 10} more lines\n")
            
        except Exception as e:
            self.explorer_response_text.insert(tk.END, f"\n⚠️ Error parsing granaries: {str(e)}\n")
            self.explorer_response_text.insert(tk.END, f"Raw output lines: {output_lines[:5]}\n")

    def get_silos_for_granary(self):
        """Get silos for the selected granary"""
        selected_granary_name = self.granary_selection_var.get()
        
        if not selected_granary_name:
            messagebox.showerror("Error", "Please select a granary first")
            return
        
        # Find the granary ID for the selected name
        selected_granary_id = None
        for name, granary_id in self.granaries_data:
            if name == selected_granary_name:
                selected_granary_id = granary_id
                break
        
        if not selected_granary_id:
            messagebox.showerror("Error", f"Could not find ID for granary: {selected_granary_name}")
            return
        
        config_file = self.explorer_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
        
        # Clear response area
        self.explorer_response_text.delete(1.0, tk.END)
        self.explorer_response_text.insert(tk.END, f"📦 Getting Silos for Granary: {selected_granary_name} (ID: {selected_granary_id})\n")
        self.explorer_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.explorer_response_text.insert(tk.END, "-" * 60 + "\n\n")
        
        # Build command using a new script or modify existing one
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "database" / "get_silos_for_granary.py"
        cmd = ["python", str(script_path), "--config", config_file, "--granary", selected_granary_id]
        
        self.explorer_response_text.insert(tk.END, f"Running command: {' '.join(cmd)}\n\n")
        self.explorer_status_var.set(f"Getting silos for {selected_granary_name}...")
        
        thread = threading.Thread(target=self._run_get_silos_subprocess, args=(cmd, selected_granary_name))
        thread.daemon = True
        thread.start()

    def _run_get_silos_subprocess(self, cmd, granary_name):
        """Execute the get silos command and populate dropdown."""
        try:
            from utils.database_utils import SubprocessUtils
            
            success, error_msg, output_lines = SubprocessUtils.run_subprocess(cmd, f"get silos for {granary_name}")
            
            # Update GUI with output in real-time
            for line in output_lines:
                self.root.after(0, self.explorer_response_text.insert, tk.END, line)
                self.root.after(0, self.explorer_response_text.see, tk.END)
            
            self.root.after(0, self.explorer_status_var.set, "")
            
            if success:
                # Parse output to extract silo names and populate dropdown
                self.root.after(0, self._parse_silos_output, output_lines, granary_name)
                self.root.after(0, self.explorer_status_var.set, f"Silos loaded for {granary_name}")
                self.root.after(0, self.explorer_response_text.insert, tk.END, f"\n✅ Silos loaded for {granary_name}!\n")
            else:
                self.root.after(0, self.explorer_status_var.set, f"Operation failed: {error_msg}")
                self.root.after(0, self.explorer_response_text.insert, tk.END, f"\n❌ Operation failed: {error_msg}\n")
                
        except Exception as exc:
            self.root.after(0, self._show_explorer_error, "Explorer Error", str(exc))

    def _parse_silos_output(self, output_lines, granary_name):
        """Parse the silos output and populate the dropdown."""
        try:
            silo_names = []
            for line in output_lines:
                # Look for lines that contain silo names
                if 'Silo:' in line:
                    # Extract silo name from line like "  Silo: SiloName (ID: xxx):"
                    parts = line.split('Silo:')
                    if len(parts) > 1:
                        silo_part = parts[1].split('(ID:')[0].strip()
                        if silo_part and silo_part not in silo_names:
                            silo_names.append(silo_part)
                # Also look for other formats that might contain silo names
                elif 'silo_name' in line.lower() or 'store_name' in line.lower():
                    # Look for table-like output
                    if '|' in line and not line.startswith('-') and not line.startswith('ID'):
                        parts = line.split('|')
                        if len(parts) >= 2:
                            silo_name = parts[1].strip()  # Silo name might be in second column
                            if silo_name and silo_name != 'silo_name' and silo_name != 'store_name':
                                silo_names.append(silo_name)
            
            # Store silo data and populate dropdown
            self.silos_data = silo_names
            self.silo_combo['values'] = silo_names
            if silo_names:
                self.silo_combo.set(silo_names[0])  # Set first silo as default
            
            self.explorer_response_text.insert(tk.END, f"\n📦 Found {len(silo_names)} silos for {granary_name}\n")
            if silo_names:
                self.explorer_response_text.insert(tk.END, f"Silos: {', '.join(silo_names[:5])}")
                if len(silo_names) > 5:
                    self.explorer_response_text.insert(tk.END, f" and {len(silo_names) - 5} more...\n")
                else:
                    self.explorer_response_text.insert(tk.END, "\n")
            else:
                self.explorer_response_text.insert(tk.END, "No silos found in output\n")
                self.explorer_response_text.insert(tk.END, "\n🔍 DEBUG: Raw output for troubleshooting:\n")
                self.explorer_response_text.insert(tk.END, "-" * 40 + "\n")
                for i, line in enumerate(output_lines[:10]):
                    self.explorer_response_text.insert(tk.END, f"{i+1:2d}: {repr(line)}\n")
                if len(output_lines) > 10:
                    self.explorer_response_text.insert(tk.END, f"... and {len(output_lines) - 10} more lines\n")
            
        except Exception as e:
            self.explorer_response_text.insert(tk.END, f"\n⚠️ Error parsing silos: {str(e)}\n")
            self.explorer_response_text.insert(tk.END, f"Raw output lines: {output_lines[:5]}\n")

    def get_date_range_for_silo(self):
        """Get date range for the selected silo"""
        selected_silo = self.silo_selection_var.get()
        selected_granary_name = self.granary_selection_var.get()
        
        if not selected_silo:
            messagebox.showerror("Error", "Please select a silo first")
            return
        
        if not selected_granary_name:
            messagebox.showerror("Error", "Please select a granary first")
            return
        
        # Find the granary ID for the selected name
        selected_granary_id = None
        for name, granary_id in self.granaries_data:
            if name == selected_granary_name:
                selected_granary_id = granary_id
                break
        
        if not selected_granary_id:
            messagebox.showerror("Error", f"Could not find ID for granary: {selected_granary_name}")
            return
        
        config_file = self.explorer_cfg_var.get().strip()
        
        if not config_file:
            messagebox.showerror("Error", "Configuration file is required")
            return
        
        # Clear response area
        self.explorer_response_text.delete(1.0, tk.END)
        self.explorer_response_text.insert(tk.END, f"📅 Getting Date Range for Silo: {selected_silo}\n")
        self.explorer_response_text.insert(tk.END, f"Granary: {selected_granary_name} (ID: {selected_granary_id})\n")
        self.explorer_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.explorer_response_text.insert(tk.END, "-" * 60 + "\n\n")
        
        # Build command using get_date_ranges.py with specific silo filter
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "database" / "get_date_range_for_silo.py"
        cmd = ["python", str(script_path), "--config", config_file, "--granary", selected_granary_id, "--silo", selected_silo]
        
        self.explorer_status_var.set(f"Getting date range for {selected_silo}...")
        
        thread = threading.Thread(target=self._run_get_silo_date_subprocess, args=(cmd, selected_silo))
        thread.daemon = True
        thread.start()

    def _run_get_silo_date_subprocess(self, cmd, silo_name):
        """Execute the get silo date range command."""
        try:
            from utils.database_utils import SubprocessUtils
            
            success, error_msg, output_lines = SubprocessUtils.run_subprocess(cmd, f"get date range for {silo_name}")
            
            # Update GUI with output in real-time
            for line in output_lines:
                self.root.after(0, self.explorer_response_text.insert, tk.END, line)
                self.root.after(0, self.explorer_response_text.see, tk.END)
            
            self.root.after(0, self.explorer_status_var.set, "")
            
            if success:
                self.root.after(0, self.explorer_status_var.set, f"Date range retrieved for {silo_name}")
                self.root.after(0, self.explorer_response_text.insert, tk.END, f"\n✅ Date range retrieved for {silo_name}!\n")
            else:
                self.root.after(0, self.explorer_status_var.set, f"Operation failed: {error_msg}")
                self.root.after(0, self.explorer_response_text.insert, tk.END, f"\n❌ Operation failed: {error_msg}\n")
                
        except Exception as exc:
            self.root.after(0, self._show_explorer_error, "Explorer Error", str(exc))

    # Remote Client Testing Methods
    def test_remote_connection(self):
        """Test connection to remote service"""
        remote_url = self.remote_url_var.get().strip()
        if not remote_url:
            messagebox.showerror("Error", "Please enter a remote service URL")
            return
        
        self.remote_status_var.set("Testing remote connection...")
        self.remote_response_text.delete(1.0, tk.END)
        self.remote_response_text.insert(tk.END, f"Testing connection to: {remote_url}\n")
        self.remote_response_text.insert(tk.END, "-" * 50 + "\n")
        
        thread = threading.Thread(target=self._test_remote_connection_thread, args=(remote_url,))
        thread.daemon = True
        thread.start()
    
    def _test_remote_connection_thread(self, remote_url):
        """Test remote connection in background thread"""
        try:
            response = requests.get(f"{remote_url}/health", timeout=10)
            if response.status_code == 200:
                self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Connection successful!\n")
                self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {response.json()}\n")
                self.root.after(0, self.remote_status_var.set, "Remote connection successful")
            else:
                self.root.after(0, self.remote_response_text.insert, tk.END, f"❌ Connection failed: HTTP {response.status_code}\n")
                self.root.after(0, self.remote_status_var.set, f"Connection failed: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.root.after(0, self.remote_response_text.insert, tk.END, f"❌ Connection error: {str(e)}\n")
            self.root.after(0, self.remote_status_var.set, f"Connection error: {str(e)}")
    
    def browse_remote_file(self):
        """Browse for file to send to remote service"""
        file_path = filedialog.askopenfilename(
            title="Select data file for remote testing",
            filetypes=[
                ("Parquet files", "*.parquet"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.remote_file_var.set(file_path)
    
    def test_remote_endpoint(self, endpoint):
        """Test a specific remote endpoint"""
        remote_url = self.remote_url_var.get().strip()
        if not remote_url:
            messagebox.showerror("Error", "Please enter a remote service URL")
            return
        
        self.remote_status_var.set(f"Testing {endpoint}...")
        self.remote_response_text.delete(1.0, tk.END)
        self.remote_response_text.insert(tk.END, f"Testing endpoint: {endpoint}\n")
        self.remote_response_text.insert(tk.END, f"URL: {remote_url}{endpoint}\n")
        self.remote_response_text.insert(tk.END, "-" * 50 + "\n")
        
        thread = threading.Thread(target=self._test_remote_endpoint_thread, args=(remote_url, endpoint))
        thread.daemon = True
        thread.start()
    
    def _test_remote_endpoint_thread(self, remote_url, endpoint):
        """Test remote endpoint in background thread"""
        try:
            if endpoint in ["/health", "/models"]:
                response = requests.get(f"{remote_url}{endpoint}", timeout=30)
            else:
                response = requests.get(f"{remote_url}{endpoint}", timeout=30)
            
            self.root.after(0, self.remote_response_text.insert, tk.END, f"Status Code: {response.status_code}\n")
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Success!\n")
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {json.dumps(json_response, indent=2)}\n")
                except:
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Success!\n")
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {response.text}\n")
                self.root.after(0, self.remote_status_var.set, f"{endpoint} test successful")
            else:
                self.root.after(0, self.remote_response_text.insert, tk.END, f"❌ Failed: HTTP {response.status_code}\n")
                self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {response.text}\n")
                self.root.after(0, self.remote_status_var.set, f"{endpoint} test failed")
        except requests.exceptions.RequestException as e:
            self.root.after(0, self.remote_response_text.insert, tk.END, f"❌ Error: {str(e)}\n")
            self.root.after(0, self.remote_status_var.set, f"{endpoint} test error")
    
    def send_remote_request(self):
        """Send file to remote service"""
        remote_url = self.remote_url_var.get().strip()
        file_path = self.remote_file_var.get().strip()
        endpoint = self.remote_endpoint_var.get().strip()
        
        if not remote_url:
            messagebox.showerror("Error", "Please enter a remote service URL")
            return
        
        if not file_path:
            messagebox.showerror("Error", "Please select a data file")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected file does not exist")
            return
        
        self.remote_status_var.set(f"Sending file to {endpoint}...")
        self.remote_response_text.delete(1.0, tk.END)
        self.remote_response_text.insert(tk.END, f"Sending file to remote service\n")
        self.remote_response_text.insert(tk.END, f"URL: {remote_url}{endpoint}\n")
        self.remote_response_text.insert(tk.END, f"File: {file_path}\n")
        self.remote_response_text.insert(tk.END, "-" * 50 + "\n")
        
        thread = threading.Thread(target=self._send_remote_request_thread, args=(remote_url, endpoint, file_path))
        thread.daemon = True
        thread.start()
    
    def _send_remote_request_thread(self, remote_url, endpoint, file_path):
        """Send remote request in background thread"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                response = requests.post(f"{remote_url}{endpoint}", files=files, timeout=300)
            
            self.root.after(0, self.remote_response_text.insert, tk.END, f"Status Code: {response.status_code}\n")
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Success!\n")
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {json.dumps(json_response, indent=2)}\n")
                except:
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Success!\n")
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {response.text}\n")
                self.root.after(0, self.remote_status_var.set, f"File sent successfully to {endpoint}")
            else:
                self.root.after(0, self.remote_response_text.insert, tk.END, f"❌ Failed: HTTP {response.status_code}\n")
                self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {response.text}\n")
                self.root.after(0, self.remote_status_var.set, f"File send failed")
        except requests.exceptions.RequestException as e:
            self.root.after(0, self.remote_response_text.insert, tk.END, f"❌ Error: {str(e)}\n")
            self.root.after(0, self.remote_status_var.set, f"Request error")
    
    def run_remote_test_suite(self):
        """Run comprehensive test suite on remote service"""
        remote_url = self.remote_url_var.get().strip()
        if not remote_url:
            messagebox.showerror("Error", "Please enter a remote service URL")
            return
        
        self.remote_status_var.set("Running full test suite...")
        self.remote_response_text.delete(1.0, tk.END)
        self.remote_response_text.insert(tk.END, f"Running comprehensive test suite\n")
        self.remote_response_text.insert(tk.END, f"Target: {remote_url}\n")
        self.remote_response_text.insert(tk.END, "=" * 60 + "\n\n")
        
        thread = threading.Thread(target=self._run_remote_test_suite_thread, args=(remote_url,))
        thread.daemon = True
        thread.start()
    
    def _run_remote_test_suite_thread(self, remote_url):
        """Run remote test suite in background thread"""
        endpoints = ["/health", "/models", "/pipeline", "/process", "/train", "/forecast"]
        results = {}
        
        for endpoint in endpoints:
            self.root.after(0, self.remote_response_text.insert, tk.END, f"Testing {endpoint}...\n")
            
            try:
                if endpoint in ["/health", "/models"]:
                    response = requests.get(f"{remote_url}{endpoint}", timeout=30)
                else:
                    response = requests.get(f"{remote_url}{endpoint}", timeout=30)
                
                if response.status_code == 200:
                    results[endpoint] = "✅ PASS"
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"  {endpoint}: ✅ PASS\n")
                else:
                    results[endpoint] = f"❌ FAIL (HTTP {response.status_code})"
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"  {endpoint}: ❌ FAIL (HTTP {response.status_code})\n")
            except Exception as e:
                results[endpoint] = f"❌ ERROR ({str(e)})"
                self.root.after(0, self.remote_response_text.insert, tk.END, f"  {endpoint}: ❌ ERROR ({str(e)})\n")
        
        # Summary
        self.root.after(0, self.remote_response_text.insert, tk.END, "\n" + "=" * 60 + "\n")
        self.root.after(0, self.remote_response_text.insert, tk.END, "TEST SUITE SUMMARY:\n")
        self.root.after(0, self.remote_response_text.insert, tk.END, "=" * 60 + "\n")
        
        passed = sum(1 for result in results.values() if "✅" in result)
        total = len(results)
        
        for endpoint, result in results.items():
            self.root.after(0, self.remote_response_text.insert, tk.END, f"{endpoint}: {result}\n")
        
        self.root.after(0, self.remote_response_text.insert, tk.END, f"\nOverall: {passed}/{total} endpoints passed\n")
        self.root.after(0, self.remote_status_var.set, f"Test suite completed: {passed}/{total} passed")
    
    def generate_remote_test_report(self):
        """Generate a test report for remote service"""
        remote_url = self.remote_url_var.get().strip()
        if not remote_url:
            messagebox.showerror("Error", "Please enter a remote service URL")
            return
        
        # Create a simple test report
        report = f"""
REMOTE SERVICE TEST REPORT
==========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target URL: {remote_url}

ENDPOINT STATUS:
- /health: Health check endpoint
- /models: Available models endpoint  
- /pipeline: Full pipeline processing
- /process: Data ingestion and preprocessing
- /train: Model training endpoint
- /forecast: Forecasting endpoint

USAGE INSTRUCTIONS:
1. Test connection first using the "Test Connection" button
2. Use individual endpoint tests for specific functionality
3. Run full test suite for comprehensive testing
4. Send data files using the file upload functionality

NETWORK REQUIREMENTS:
- Ensure firewall allows connections to port 8000
- Verify network connectivity between client and server
- Check that the remote service is running and accessible
"""
        
        self.remote_response_text.delete(1.0, tk.END)
        self.remote_response_text.insert(tk.END, report)
        self.remote_status_var.set("Test report generated")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SiloFlowTester(root)
    
    # Add some helpful instructions to the logs tab
    app.logs_text.insert(tk.END, "📋 Quick Start Guide:\n")
    app.logs_text.insert(tk.END, "=" * 50 + "\n\n")
    app.logs_text.insert(tk.END, "🌐 HTTP Service Testing:\n")
    app.logs_text.insert(tk.END, "1. Go to 'HTTP Service Testing' tab\n")
    app.logs_text.insert(tk.END, "2. Use Local/Remote buttons to switch between services\n")
    app.logs_text.insert(tk.END, "3. Test connection to verify service is running\n")
    app.logs_text.insert(tk.END, "4. Select a data file and endpoint\n")
    app.logs_text.insert(tk.END, "5. Send request to test the API\n\n")
    
    app.logs_text.insert(tk.END, "🌍 Remote Client Testing:\n")
    app.logs_text.insert(tk.END, "1. Go to 'Remote Client Testing' tab\n")
    app.logs_text.insert(tk.END, "2. Configure remote service URL (use get_my_ip.py to find your IP)\n")
    app.logs_text.insert(tk.END, "3. Test connection to remote service\n")
    app.logs_text.insert(tk.END, "4. Run individual endpoint tests or full test suite\n")
    app.logs_text.insert(tk.END, "5. Send data files to remote service\n\n")
    
    app.logs_text.insert(tk.END, "🗄️ Data Retrieval:\n")
    app.logs_text.insert(tk.END, "1. Go to 'Data Retrieval' tab\n")
    app.logs_text.insert(tk.END, "2. Configure database settings\n")
    app.logs_text.insert(tk.END, "3. Choose retrieval mode and options\n")
    app.logs_text.insert(tk.END, "4. Run automated data retrieval\n\n")
    
    app.logs_text.insert(tk.END, "🔍 Database Explorer:\n")
    app.logs_text.insert(tk.END, "1. Go to 'Database Explorer' tab\n")
    app.logs_text.insert(tk.END, "2. Test database connection first\n")
    app.logs_text.insert(tk.END, "3. List granaries and silos\n")
    app.logs_text.insert(tk.END, "4. Get date ranges for planning\n\n")
    
    app.logs_text.insert(tk.END, "📊 All operations show real-time progress and results.\n")
    app.logs_text.insert(tk.END, "📁 Check the response areas in each tab for detailed output.\n\n")
    
    root.mainloop()

if __name__ == "__main__":
    import sys
    main()
