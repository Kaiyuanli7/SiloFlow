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
import time
import threading
import asyncio
from datetime import datetime
import subprocess
import re # Added for regex in _parse_granaries_output

class SiloFlowTester:
    def __init__(self, root):
        self.root = root
        self.root.title("🌾 SiloFlow - Advanced Testing & Data Management Interface")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Ensure window is resizable
        self.root.resizable(True, True)
        
        # Configure window icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Configure modern styling
        self.setup_modern_style()
        
        # Service configuration
        self.service_url = "http://localhost:8000"
        self.remote_service_url = "http://192.168.28.242:8000"  # Default remote URL
        
        # Initialize data storage
        self.retrieval_granaries_data = []  # List of (name, id) tuples for retrieval dropdown
        self.all_silos_data = []  # Store all silo data for auto-fill functionality
        self.current_silo_index = 0  # Track current silo for auto-fill
        
        # Color scheme
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Accent purple
            'success': '#F18F01',      # Warm orange
            'warning': '#C73E1D',      # Red warning
            'background': '#F5F5F5',   # Light gray background
            'surface': '#FFFFFF',      # White surface
            'text_primary': '#000000', # Black text for visibility
            'text_secondary': '#333333' # Dark gray text
        }
        
        # Configure root background
        self.root.configure(bg=self.colors['background'])
        
        # Create GUI elements
        self.create_widgets()
        
        # Center window on screen
        self.center_window()
        
    def get_python_executable(self):
        """Get the correct Python executable path for subprocess calls"""
        # The script is in: g:\liky\siloflow\service\scripts\testing\testingservice.py
        # The .venv is in: g:\liky\siloflow\.venv\Scripts\python.exe
        # So we need to go up 3 levels: testing -> scripts -> service -> siloflow
        
        script_dir = Path(__file__).parent  # testing/
        scripts_dir = script_dir.parent     # scripts/
        service_dir = scripts_dir.parent    # service/
        siloflow_root = service_dir.parent  # siloflow/
        
        venv_path = siloflow_root / ".venv" / "Scripts" / "python.exe"
        
        if venv_path.exists():
            return str(venv_path)
        
        # Fallback to system python
        return sys.executable
        
    def setup_modern_style(self):
        """Configure modern ttk styling"""
        self.style = ttk.Style()
        
        # Use a modern theme as base
        available_themes = self.style.theme_names()
        if 'winnative' in available_themes:
            self.style.theme_use('winnative')
        elif 'clam' in available_themes:
            self.style.theme_use('clam')
        else:
            self.style.theme_use('default')
        
        # Configure custom styles
        self.configure_custom_styles()
        
    def configure_custom_styles(self):
        """Configure custom ttk styles for modern appearance"""
        # Primary button style
        self.style.configure(
            'Primary.TButton',
            background='#2E86AB',
            foreground='black',
            borderwidth=0,
            focuscolor='none',
            font=('Segoe UI', 9, 'bold'),
            padding=(10, 8)
        )
        self.style.map(
            'Primary.TButton',
            background=[('active', '#1E5F7A'), ('pressed', '#1A4F66')]
        )
        
        # Success button style
        self.style.configure(
            'Success.TButton',
            background='#F18F01',
            foreground='black',
            borderwidth=0,
            focuscolor='none',
            font=('Segoe UI', 9, 'bold'),
            padding=(10, 8)
        )
        self.style.map(
            'Success.TButton',
            background=[('active', '#D67A01'), ('pressed', '#BC6A01')]
        )
        
        # Warning button style
        self.style.configure(
            'Warning.TButton',
            background='#C73E1D',
            foreground='black',
            borderwidth=0,
            focuscolor='none',
            font=('Segoe UI', 9, 'bold'),
            padding=(10, 8)
        )
        self.style.map(
            'Warning.TButton',
            background=[('active', '#A33218'), ('pressed', '#8A2B15')]
        )
        
        # Modern notebook style
        self.style.configure(
            'Modern.TNotebook',
            background='#F5F5F5',
            borderwidth=0,
            tabposition='n'
        )
        self.style.configure(
            'Modern.TNotebook.Tab',
            background='#E2E8F0',
            foreground='#2D3748',
            padding=[20, 10],
            font=('Segoe UI', 10, 'bold')
        )
        self.style.map(
            'Modern.TNotebook.Tab',
            background=[('selected', '#FFFFFF'), ('active', '#F7FAFC')],
            foreground=[('selected', '#2E86AB')]
        )
        
        # Modern labelframe style
        self.style.configure(
            'Modern.TLabelframe',
            background='#FFFFFF',
            borderwidth=1,
            relief='solid'
        )
        self.style.configure(
            'Modern.TLabelframe.Label',
            background='#FFFFFF',
            foreground='#2E86AB',
            font=('Segoe UI', 10, 'bold')
        )
        
        # Modern entry style
        self.style.configure(
            'Modern.TEntry',
            borderwidth=1,
            relief='solid',
            fieldbackground='#FFFFFF',
            font=('Segoe UI', 9),
            padding=(8, 6)
        )
        
        # Modern combobox style
        self.style.configure(
            'Modern.TCombobox',
            borderwidth=1,
            relief='solid',
            fieldbackground='#FFFFFF',
            font=('Segoe UI', 9),
            padding=(8, 6)
        )
        
        # Default ttk Label style to ensure text visibility
        self.style.configure(
            'TLabel',
            foreground='#000000',  # Black text
            font=('Segoe UI', 9)
        )
        
        # Title label style
        self.style.configure(
            'Title.TLabel',
            foreground='#000000',  # Black text
            font=('Segoe UI', 14, 'bold')
        )
        
        # Enhanced button styles to ensure visibility
        self.style.configure(
            'TButton',
            foreground='#000000',  # Black text as fallback
            font=('Segoe UI', 9)
        )
        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def create_modern_header(self, parent):
        """Create a modern header with title and status"""
        header_frame = tk.Frame(parent, bg=self.colors['primary'], height=60)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🌾 SiloFlow Testing & Management Interface",
            bg=self.colors['primary'],
            fg='white',
            font=('Segoe UI', 16, 'bold'),
            anchor='w'
        )
        title_label.pack(side='left', padx=20, pady=15)
        
        # Status indicator
        self.connection_status = tk.Label(
            header_frame,
            text="🔴 Disconnected",
            bg=self.colors['primary'],
            fg='white',
            font=('Segoe UI', 10),
            anchor='e'
        )
        self.connection_status.pack(side='right', padx=20, pady=15)
        
        return header_frame
    
    def create_scrollable_frame(self, parent):
        """Create a scrollable frame for tabs with lots of content"""
        # Create main frame that will contain the canvas and scrollbar
        main_frame = tk.Frame(parent, bg=self.colors['background'])
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame, bg=self.colors['background'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        # Create the scrollable frame that will contain all the content
        scrollable_frame = tk.Frame(canvas, bg=self.colors['background'])
        
        # Configure scroll region when frame changes size
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind('<Configure>', configure_scroll_region)
        
        # Create window in canvas
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configure canvas width when window changes
        def configure_canvas_width(event):
            canvas.itemconfig(canvas_frame, width=event.width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas for scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        # Bind mouse wheel events when entering/leaving the canvas
        canvas.bind('<Enter>', bind_mousewheel)
        canvas.bind('<Leave>', unbind_mousewheel)
        
        return main_frame, scrollable_frame
        
    def create_widgets(self):
        # Create main container
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill='both', expand=True)
        
        # Create modern header
        self.create_modern_header(main_container)
        
        # Create notebook for tabs with modern styling
        self.notebook = ttk.Notebook(main_container, style='Modern.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Create tabs with enhanced styling
        self.create_http_service_tab()
        self.create_remote_client_tab()
        self.create_simple_retrieval_tab()
        self.create_batch_processing_tab()
        self.create_database_explorer_tab()
        # Removed logs tab as requested
        
        # Bind tab change event for status updates
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
    def on_tab_changed(self, event):
        """Handle tab change events for dynamic updates"""
        try:
            selected_tab = self.notebook.tab(self.notebook.select(), "text")
            # Update status based on selected tab
            if "HTTP Service" in selected_tab:
                self.update_connection_status()
        except:
            pass
            
    def update_connection_status(self):
        """Update the connection status indicator asynchronously"""
        def check_status():
            try:
                response = requests.get(f"{self.service_url}/health", timeout=1)
                if response.status_code == 200:
                    self.root.after(0, lambda: self.connection_status.config(text="🟢 Connected", fg='#10B981'))
                else:
                    self.root.after(0, lambda: self.connection_status.config(text="🟡 Limited", fg='#F59E0B'))
            except:
                self.root.after(0, lambda: self.connection_status.config(text="🔴 Disconnected", fg='#EF4444'))
        
        # Run the status check in a background thread to avoid freezing GUI
        import threading
        threading.Thread(target=check_status, daemon=True).start()
    
    def create_section_frame(self, parent, title, icon=""):
        """Create a modern section frame with consistent styling"""
        frame = ttk.LabelFrame(
            parent, 
            text=f"{icon} {title}" if icon else title,
            style='Modern.TLabelframe',
            padding="15"
        )
        return frame
        
    def create_modern_button(self, parent, text, command, style="Primary.TButton", **kwargs):
        """Create a modern styled button with ensured text visibility"""
        try:
            # Try to create TTK button with style
            button = ttk.Button(
                parent,
                text=text,
                command=command,
                style=style,
                **kwargs
            )
            # Force black text for visibility - override any white text issues
            if style == "Primary.TButton":
                self.style.configure(style, foreground='black', background='#2E86AB')
            elif style == "Success.TButton":
                self.style.configure(style, foreground='black', background='#F18F01')
            elif style == "Warning.TButton":
                self.style.configure(style, foreground='black', background='#C73E1D')
            else:
                self.style.configure(style, foreground='black')
            return button
        except:
            # Fallback to regular tk.Button if TTK fails - also with black text
            colors = {
                "Primary.TButton": {'bg': '#2E86AB', 'fg': 'black'},
                "Success.TButton": {'bg': '#F18F01', 'fg': 'black'},
                "Warning.TButton": {'bg': '#C73E1D', 'fg': 'black'},
            }
            button_colors = colors.get(style, {'bg': '#2E86AB', 'fg': 'black'})
            return tk.Button(
                parent,
                text=text,
                command=command,
                bg=button_colors['bg'],
                fg=button_colors['fg'],
                font=('Segoe UI', 9, 'bold'),
                relief='flat',
                borderwidth=0,
                padx=10,
                pady=8,
                **kwargs
            )
        
    def create_http_service_tab(self):
        """Create HTTP Service Testing tab with modern styling"""
        http_frame = ttk.Frame(self.notebook)
        self.notebook.add(http_frame, text="🌐 HTTP Service Testing")
        
        # Configure grid
        http_frame.columnconfigure(1, weight=1)
        http_frame.rowconfigure(4, weight=1)
        
        # Service Configuration Section
        config_section = self.create_section_frame(http_frame, "Service Configuration", "⚙️")
        config_section.grid(row=0, column=0, columnspan=3, sticky="ew", pady=10, padx=15)
        config_section.columnconfigure(1, weight=1)
        
        # Service URL with quick buttons
        tk.Label(config_section, text="Service URL:", font=('Segoe UI', 10, 'bold'), 
                fg=self.colors['text_primary'], bg='white').grid(row=0, column=0, sticky="w", pady=8)
        self.url_var = tk.StringVar(value=self.service_url)
        url_entry = ttk.Entry(config_section, textvariable=self.url_var, font=('Segoe UI', 10), style='Modern.TEntry')
        url_entry.grid(row=0, column=1, sticky="ew", pady=8, padx=(10, 0))
        
        # Quick URL buttons with modern styling
        url_buttons_frame = tk.Frame(config_section, bg='white')
        url_buttons_frame.grid(row=0, column=2, padx=(10, 0), pady=8)
        
        self.create_modern_button(url_buttons_frame, "Local", 
                                 lambda: self.url_var.set("http://localhost:8000"), 
                                 "Primary.TButton").pack(side=tk.LEFT, padx=(0, 5))
        self.create_modern_button(url_buttons_frame, "Remote", 
                                 lambda: self.url_var.set(self.remote_service_url),
                                 "Success.TButton").pack(side=tk.LEFT, padx=(0, 5))
        self.create_modern_button(url_buttons_frame, "🔍 Test", 
                                 self.test_connection,
                                 "Warning.TButton").pack(side=tk.LEFT)
        
        # File Selection Section
        file_section = self.create_section_frame(http_frame, "File Selection", "📁")
        file_section.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10, padx=15)
        file_section.columnconfigure(1, weight=1)
        
        tk.Label(file_section, text="Data File:", font=('Segoe UI', 10, 'bold'), 
                fg=self.colors['text_primary'], bg='white').grid(row=0, column=0, sticky="w", pady=8)
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_section, textvariable=self.file_var, font=('Segoe UI', 10), style='Modern.TEntry')
        file_entry.grid(row=0, column=1, sticky="ew", pady=8, padx=(10, 0))
        self.create_modern_button(file_section, "📂 Browse", self.browse_file, "Primary.TButton").grid(row=0, column=2, padx=(10, 0), pady=8)
        
        # File info display
        self.file_info_var = tk.StringVar(value="No file selected")
        file_info_label = tk.Label(file_section, textvariable=self.file_info_var, 
                                  font=('Segoe UI', 9), fg=self.colors['text_secondary'], bg='white')
        file_info_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))
        
        # Endpoint Selection Section
        endpoint_section = self.create_section_frame(http_frame, "Endpoint Configuration", "🎯")
        endpoint_section.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10, padx=15)
        endpoint_section.columnconfigure(1, weight=1)
        
        tk.Label(endpoint_section, text="Endpoint:", font=('Segoe UI', 10, 'bold'), 
                fg=self.colors['text_primary'], bg='white').grid(row=0, column=0, sticky="w", pady=8)
        self.endpoint_var = tk.StringVar(value="/pipeline")
        endpoint_combo = ttk.Combobox(
            endpoint_section,
            textvariable=self.endpoint_var,
            values=["/health", "/models", "/sort", "/process", "/train", "/forecast", "/pipeline"],
            state="readonly",
            font=('Segoe UI', 10),
            style='Modern.TCombobox'
        )
        endpoint_combo.grid(row=0, column=1, sticky="w", pady=8, padx=(10, 0))
        
        # Send button with enhanced styling
        send_button = self.create_modern_button(endpoint_section, "🚀 Send Request", self.send_request, "Success.TButton")
        send_button.grid(row=0, column=2, padx=(20, 0), pady=8)
        
        # Response Section
        response_section = self.create_section_frame(http_frame, "Response & Results", "📊")
        response_section.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=10, padx=15)
        response_section.columnconfigure(0, weight=1)
        response_section.rowconfigure(0, weight=1)
        
        # Response text with modern styling
        self.http_response_text = scrolledtext.ScrolledText(
            response_section, 
            height=15, 
            width=80,
            font=('Consolas', 9),
            bg='#F8F9FA',
            fg=self.colors['text_primary'],
            selectbackground=self.colors['primary'],
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        self.http_response_text.grid(row=0, column=0, sticky="nsew", pady=5)
        
        # Status bar with modern styling
        status_frame = tk.Frame(http_frame, bg=self.colors['surface'], height=35)
        status_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=15, pady=(0, 10))
        status_frame.pack_propagate(False)
        
        self.http_status_var = tk.StringVar(value="Ready - Select a file and endpoint to begin")
        status_label = tk.Label(
            status_frame,
            textvariable=self.http_status_var,
            bg=self.colors['surface'],
            fg=self.colors['text_secondary'],
            font=('Segoe UI', 9),
            anchor='w'
        )
        status_label.pack(side='left', padx=15, pady=8)
        
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
            values=["/health", "/models", "/sort", "/process", "/train", "/forecast", "/pipeline"],
            state="readonly",
            width=20
        )
        remote_endpoint_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Quick test buttons
        quick_test_frame = ttk.Frame(endpoint_frame)
        quick_test_frame.grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(quick_test_frame, text="Test /health", command=lambda: self.test_remote_endpoint("/health")).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_test_frame, text="Test /models", command=lambda: self.test_remote_endpoint("/models")).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_test_frame, text="Test /sort", command=lambda: self.test_remote_endpoint("/sort")).pack(side=tk.LEFT, padx=2)
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
        """Create Production Pipeline Data Retrieval tab with scrolling capability"""
        # Create the main tab frame
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🚀 Production Pipeline")
        
        # Create scrollable frame for this tab
        main_frame, retrieval_frame = self.create_scrollable_frame(tab_frame)
        main_frame.pack(fill='both', expand=True)
        
        # Configure the scrollable content area
        retrieval_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(retrieval_frame, text="Production Data Pipeline - Enterprise Scale", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Configuration section
        config_frame = ttk.LabelFrame(retrieval_frame, text="Pipeline Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="Config File:").grid(row=0, column=0, sticky="w", pady=2)
        # Set default config path to the production config
        default_config = str(Path(__file__).parent.parent.parent / "config" / "production_config.json")
        self.retrieval_cfg_var = tk.StringVar(value=default_config)
        cfg_entry = ttk.Entry(config_frame, textvariable=self.retrieval_cfg_var, width=50)
        cfg_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self._browse_retrieval_cfg).grid(row=0, column=2, padx=5, pady=2)
        
        # Pipeline phases section
        phases_frame = ttk.LabelFrame(retrieval_frame, text="Pipeline Phases", padding="10")
        phases_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
        # Checkboxes for pipeline phases
        self.run_retrieval_var = tk.BooleanVar(value=True)
        self.run_preprocessing_var = tk.BooleanVar(value=True)
        self.run_training_var = tk.BooleanVar(value=True)
        self.cleanup_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(phases_frame, text="Data Retrieval (Stream from database)", variable=self.run_retrieval_var).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Checkbutton(phases_frame, text="Data Preprocessing (Clean & feature engineering)", variable=self.run_preprocessing_var).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Checkbutton(phases_frame, text="Model Training (Train forecasting models)", variable=self.run_training_var).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Checkbutton(phases_frame, text="Cleanup temp files after completion", variable=self.cleanup_var).grid(row=3, column=0, sticky="w", pady=2)
        
        # Retrieval options section
        options_frame = ttk.LabelFrame(retrieval_frame, text="Retrieval Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        options_frame.columnconfigure(1, weight=1)
        
        # Retrieval mode
        ttk.Label(options_frame, text="Mode:").grid(row=0, column=0, sticky="w", pady=2)
        self.retrieval_mode_var = tk.StringVar(value="full-retrieval")
        mode_combo = ttk.Combobox(
            options_frame,
            textvariable=self.retrieval_mode_var,
            values=["full-retrieval", "incremental", "date-range"],
            state="readonly",
            width=15
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        mode_combo.bind('<<ComboboxSelected>>', self._on_mode_change)
        
        # Granary selection
        ttk.Label(options_frame, text="Granary:").grid(row=1, column=0, sticky="w", pady=2)
        self.retrieval_granary_var = tk.StringVar()
        self.retrieval_granary_combo = ttk.Combobox(options_frame, textvariable=self.retrieval_granary_var, width=30, state="readonly")
        self.retrieval_granary_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(options_frame, text="📋 Load", command=self.load_granaries_for_retrieval).grid(row=1, column=2, padx=5, pady=2)
        
        # Performance options
        perf_frame = ttk.Frame(options_frame)
        perf_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(perf_frame, text="Max Records/Granary (testing):").grid(row=0, column=0, sticky="w", pady=2)
        self.max_records_var = tk.StringVar()
        ttk.Entry(perf_frame, textvariable=self.max_records_var, width=15).grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(perf_frame, text="Days (incremental):").grid(row=0, column=2, sticky="w", pady=2, padx=(20,0))
        self.days_var = tk.StringVar(value="7")
        ttk.Entry(perf_frame, textvariable=self.days_var, width=10).grid(row=0, column=3, sticky="w", padx=5)
        
        # Date range (initially hidden)
        self.date_range_frame = ttk.Frame(options_frame)
        self.date_range_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        self.date_range_frame.grid_remove()  # Hidden by default
        
        ttk.Label(self.date_range_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="w")
        self.start_date_var = tk.StringVar()
        ttk.Entry(self.date_range_frame, textvariable=self.start_date_var, width=15).grid(row=0, column=1, sticky="w", padx=2)
        
        ttk.Label(self.date_range_frame, text="End Date (YYYY-MM-DD):").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.end_date_var = tk.StringVar()
        ttk.Entry(self.date_range_frame, textvariable=self.end_date_var, width=15).grid(row=0, column=3, sticky="w", padx=2)
        
        # System monitoring section
        monitor_frame = ttk.LabelFrame(retrieval_frame, text="System Monitoring", padding="10")
        monitor_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
        # System stats display
        self.system_stats_var = tk.StringVar(value="Click 'Check System' to view resource usage")
        stats_label = ttk.Label(monitor_frame, textvariable=self.system_stats_var)
        stats_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        
        ttk.Button(monitor_frame, text="Check System", command=self.check_system_resources).grid(row=0, column=2, padx=5)
        ttk.Button(monitor_frame, text="Open Task Manager", command=self.open_task_manager).grid(row=0, column=3, padx=5)
        
        # Run buttons
        buttons_frame = ttk.Frame(retrieval_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        ttk.Button(buttons_frame, text="🚀 Run Production Pipeline", 
                  command=self.run_production_pipeline, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="⚡ Quick Test (10K records)", 
                  command=self.run_quick_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="🔄 Check Status", 
                  command=self.check_pipeline_status).pack(side=tk.LEFT, padx=5)
        
        # Response area
        response_frame = ttk.LabelFrame(retrieval_frame, text="Pipeline Output & Monitoring", padding="10")
        response_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=5, padx=5)
        response_frame.columnconfigure(0, weight=1)
        response_frame.rowconfigure(0, weight=1)
        
        self.retrieval_response_text = scrolledtext.ScrolledText(response_frame, height=15, width=80)
        self.retrieval_response_text.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.retrieval_status_var = tk.StringVar(value="Ready - Production Pipeline")
        status_label = ttk.Label(retrieval_frame, textvariable=self.retrieval_status_var, relief=tk.SUNKEN)
        status_label.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
    def create_simple_retrieval_tab(self):
        """Create Simple Data Retrieval tab with modern styling and scrolling"""
        # Create the main tab frame
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📊 Simple Retrieval")
        
        # Create scrollable frame for this tab
        main_frame, simple_frame = self.create_scrollable_frame(tab_frame)
        main_frame.pack(fill='both', expand=True)
        
        # Configure the scrollable content area
        simple_frame.columnconfigure(0, weight=1)
        
        # Configuration Section
        config_section = self.create_section_frame(simple_frame, "Retrieval Configuration", "⚙️")
        config_section.grid(row=0, column=0, sticky="ew", pady=10, padx=15)
        config_section.columnconfigure(1, weight=1)
        
        # Granary name with modern styling
        tk.Label(config_section, text="Granary Name:", font=('Segoe UI', 10, 'bold'), 
                fg=self.colors['text_primary'], bg='white').grid(row=0, column=0, sticky="w", pady=8)
        self.simple_granary_var = tk.StringVar(value="蚬冈库")
        granary_entry = ttk.Entry(config_section, textvariable=self.simple_granary_var, 
                                 font=('Segoe UI', 10), style='Modern.TEntry', width=30)
        granary_entry.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=8)
        
        # Silo ID
        tk.Label(config_section, text="Silo ID:", font=('Segoe UI', 10, 'bold'), 
                fg=self.colors['text_primary'], bg='white').grid(row=1, column=0, sticky="w", pady=8)
        self.simple_silo_var = tk.StringVar(value="41f2257ce3d64083b1b5f8e59e80bc4d")
        silo_entry = ttk.Entry(config_section, textvariable=self.simple_silo_var, 
                              font=('Segoe UI', 10), style='Modern.TEntry')
        silo_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=8)
        
        # Date range in a neat grid
        date_container = tk.Frame(config_section, bg='white')
        date_container.grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)
        date_container.columnconfigure(1, weight=1)
        date_container.columnconfigure(3, weight=1)
        
        tk.Label(date_container, text="Start Date:", font=('Segoe UI', 10, 'bold'), bg='white', 
                fg=self.colors['text_primary']).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.simple_start_date_var = tk.StringVar(value="2024-07-17")
        start_date_entry = ttk.Entry(date_container, textvariable=self.simple_start_date_var, 
                                   font=('Segoe UI', 10), style='Modern.TEntry', width=15)
        start_date_entry.grid(row=0, column=1, sticky="w")
        
        tk.Label(date_container, text="End Date:", font=('Segoe UI', 10, 'bold'), bg='white', 
                fg=self.colors['text_primary']).grid(row=0, column=2, sticky="w", padx=(20, 10))
        self.simple_end_date_var = tk.StringVar(value="2024-07-18")
        end_date_entry = ttk.Entry(date_container, textvariable=self.simple_end_date_var, 
                                  font=('Segoe UI', 10), style='Modern.TEntry', width=15)
        end_date_entry.grid(row=0, column=3, sticky="w")
        
        # Output Configuration Section
        output_section = self.create_section_frame(simple_frame, "Output Configuration", "📁")
        output_section.grid(row=1, column=0, sticky="ew", pady=10, padx=15)
        output_section.columnconfigure(1, weight=1)
        
        tk.Label(output_section, text="Output Directory:", font=('Segoe UI', 10, 'bold'), 
                fg=self.colors['text_primary'], bg='white').grid(row=0, column=0, sticky="w", pady=8)
        self.simple_output_var = tk.StringVar(value="data/simple_retrieval")
        output_entry = ttk.Entry(output_section, textvariable=self.simple_output_var, 
                               font=('Segoe UI', 10), style='Modern.TEntry')
        output_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=8)
        self.create_modern_button(output_section, "📂 Browse", self.browse_simple_output_dir, "Primary.TButton").grid(row=0, column=2, padx=(10, 0), pady=8)
        
        # Actions Section
        actions_section = self.create_section_frame(simple_frame, "Actions", "🚀")
        actions_section.grid(row=2, column=0, sticky="ew", pady=10, padx=15)
        
        # Action buttons - Organized in a grid for better layout
        button_grid = tk.Frame(actions_section, bg=self.colors['surface'])
        button_grid.pack(fill='x', pady=5)
        
        # Row 1
        buttons_row1 = tk.Frame(button_grid, bg=self.colors['surface'])
        buttons_row1.pack(fill='x', pady=(0, 8))
        
        self.create_modern_button(buttons_row1, "🔍 Silos", self.list_all_silos, "Primary.TButton").pack(side=tk.LEFT, padx=(0, 8))
        self.create_modern_button(buttons_row1, "🏢 Get Granaries & List All Silos", self.get_granaries_and_silos, "Primary.TButton").pack(side=tk.LEFT, padx=(0, 8))
        self.create_modern_button(buttons_row1, "🔄 Auto-Fill Next Silo", self.auto_fill_next_silo, "Success.TButton").pack(side=tk.LEFT, padx=(0, 8))
        self.create_modern_button(buttons_row1, "📊 Retrieve Data", self.run_simple_retrieval, "Warning.TButton").pack(side=tk.LEFT)
        
        # Row 2
        buttons_row2 = tk.Frame(button_grid, bg=self.colors['surface'])
        buttons_row2.pack(fill='x')
        
        self.create_modern_button(buttons_row2, "🚀 Auto Process All Silos", self.auto_process_all_silos, "Success.TButton").pack(side=tk.LEFT, padx=(0, 8))
        self.create_modern_button(buttons_row2, "📂 Open Output Folder", self.open_simple_output_folder, "Primary.TButton").pack(side=tk.LEFT, padx=(0, 8))
        self.create_modern_button(buttons_row2, "🧹 Clear Log", self.clear_simple_log, "Primary.TButton").pack(side=tk.LEFT)
        
        # Progress Section
        progress_section = self.create_section_frame(simple_frame, "Progress Status", "📈")
        progress_section.grid(row=3, column=0, sticky="ew", pady=10, padx=15)
        
        self.simple_progress_var = tk.StringVar(value="Ready to begin data retrieval")
        progress_label = tk.Label(progress_section, textvariable=self.simple_progress_var,
                                font=('Segoe UI', 10), fg=self.colors['primary'], bg='white')
        progress_label.pack(fill='x', pady=8)
        
        # Response/Log Section
        log_section = self.create_section_frame(simple_frame, "Retrieval Log & Output", "📋")
        log_section.grid(row=4, column=0, sticky="nsew", pady=10, padx=15)
        log_section.columnconfigure(0, weight=1)
        log_section.rowconfigure(0, weight=1)
        
        self.simple_response_text = scrolledtext.ScrolledText(
            log_section, 
            height=15, 
            width=80,
            font=('Consolas', 9),
            bg='#F8F9FA',
            fg=self.colors['text_primary'],
            selectbackground=self.colors['primary'],
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        self.simple_response_text.grid(row=0, column=0, sticky="nsew", pady=5)
        
        # Add initial helpful message
        self.simple_response_text.insert(tk.END, "🌾 Simple Data Retrieval System\n")
        self.simple_response_text.insert(tk.END, "=" * 50 + "\n\n")
        self.simple_response_text.insert(tk.END, "📋 Quick Start Guide:\n")
        self.simple_response_text.insert(tk.END, "1. Click 'Get Granaries & Silos' to load available data sources\n")
        self.simple_response_text.insert(tk.END, "2. Use 'Auto-Fill Next Silo' to automatically populate form fields\n")
        self.simple_response_text.insert(tk.END, "3. Adjust date range as needed\n")
        self.simple_response_text.insert(tk.END, "4. Click 'Retrieve Data' to download silo data\n")
        self.simple_response_text.insert(tk.END, "5. Use 'Auto Process All Silos' for batch operations (can select CSV file)\n\n")
        self.simple_response_text.insert(tk.END, "💡 Tip: Check the output folder after each retrieval\n")
        self.simple_response_text.insert(tk.END, "🔧 All operations show real-time progress here\n\n")

    def create_database_explorer_tab(self):
        """Create Database Explorer tab with scrolling capability"""
        # Create the main tab frame
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Database Explorer")
        
        # Create scrollable frame for this tab
        main_frame, explorer_frame = self.create_scrollable_frame(tab_frame)
        main_frame.pack(fill='both', expand=True)
        
        # Configure the scrollable content area
        explorer_frame.columnconfigure(1, weight=1)
        
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
        
    def create_batch_processing_tab(self):
        """Create Batch Processing tab for folder-based operations with scrolling support"""
        # Create main frame for the tab
        main_batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_batch_frame, text="🔄 Batch Processing")
        
        # Configure main frame grid
        main_batch_frame.columnconfigure(0, weight=1)
        main_batch_frame.rowconfigure(0, weight=1)
        
        # Create a canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_batch_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_batch_frame, orient="vertical", command=canvas.yview)
        
        # Create the scrollable frame that will contain all the content
        batch_frame = ttk.Frame(canvas)
        
        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame = canvas.create_window((0, 0), window=batch_frame, anchor="nw")
        
        # Grid the canvas and scrollbar
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid for the scrollable content frame
        batch_frame.columnconfigure(1, weight=1)
        
        # Bind canvas resize to update scroll region
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Also update the canvas window width to match canvas width
            canvas_width = canvas.winfo_width()
            if canvas_width > 1:  # Avoid setting width to 1 during initial setup
                canvas.itemconfig(canvas_frame, width=canvas_width)
        
        batch_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_scroll_region)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', bind_mousewheel)
        canvas.bind('<Leave>', unbind_mousewheel)
        
        # Now create all the content in the scrollable batch_frame
        # Configure grid for content
        batch_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(batch_frame, text="Batch Processing Pipeline", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Folder selection section
        folder_frame = ttk.LabelFrame(batch_frame, text="Folder Selection", padding="10")
        folder_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="Input Folder:").grid(row=0, column=0, sticky="w", pady=2)
        self.batch_input_folder_var = tk.StringVar(value="G:/liky/siloflow/service/data/simple_retrieval")
        input_folder_entry = ttk.Entry(folder_frame, textvariable=self.batch_input_folder_var, width=60)
        input_folder_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(folder_frame, text="Browse", command=self.browse_batch_input_folder).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(folder_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=2)
        self.batch_output_folder_var = tk.StringVar(value="G:/liky/siloflow/service/data/granaries")
        output_folder_entry = ttk.Entry(folder_frame, textvariable=self.batch_output_folder_var, width=60)
        output_folder_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(folder_frame, text="Browse", command=self.browse_batch_output_folder).grid(row=1, column=2, padx=5, pady=2)
        
        # Processing action selection section
        action_frame = ttk.LabelFrame(batch_frame, text="Choose Processing Action", padding="10")
        action_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
        # Radio buttons for single action selection
        self.batch_action_var = tk.StringVar(value="sorting")
        
        ttk.Radiobutton(action_frame, text="🔤 Data Sorting Only", 
                       variable=self.batch_action_var, value="sorting").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        ttk.Radiobutton(action_frame, text="⚙️ Data Processing Only", 
                       variable=self.batch_action_var, value="processing").grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        ttk.Radiobutton(action_frame, text="🤖 Model Training Only", 
                       variable=self.batch_action_var, value="training").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        
        ttk.Radiobutton(action_frame, text="🔮 Forecasting Only", 
                       variable=self.batch_action_var, value="forecasting").grid(row=1, column=1, sticky="w", padx=10, pady=5)
        
        # Add description for each action
        description_frame = ttk.Frame(action_frame)
        description_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.action_description_var = tk.StringVar(value="Data Sorting: Organizes raw data files into standardized granary-specific .parquet files")
        description_label = ttk.Label(description_frame, textvariable=self.action_description_var, 
                                     wraplength=600, foreground="gray")
        description_label.pack(anchor="w")
        
        # Bind radio button changes to update description
        def update_description(*args):
            action = self.batch_action_var.get()
            descriptions = {
                "sorting": "Data Sorting: Organizes raw data files into standardized granary-specific .parquet files",
                "processing": "Data Processing: Cleans, validates, and enriches data with feature engineering",
                "training": "Model Training: Trains machine learning models using processed data with hyperparameter optimization",
                "forecasting": "Forecasting: Generates temperature predictions using trained models"
            }
            self.action_description_var.set(descriptions.get(action, ""))
            
            # Show/hide tuning frame based on selection
            if action == "training":
                tuning_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5, padx=10)
            else:
                tuning_frame.grid_remove()
        
        self.batch_action_var.trace('w', update_description)
        
        # Hyperparameter tuning options (sub-section) - only for training action
        tuning_frame = ttk.LabelFrame(action_frame, text="Training & Performance Options", padding="5")
        tuning_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5, padx=10)
        tuning_frame.columnconfigure(1, weight=1)
        
        # GPU acceleration option
        self.batch_gpu_var = tk.BooleanVar(value=False)
        gpu_checkbox = ttk.Checkbutton(tuning_frame, text="🚀 Enable GPU Acceleration (Training & Forecasting)", variable=self.batch_gpu_var)
        gpu_checkbox.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        
        # Add tooltip/info about GPU
        def show_gpu_info():
            import tkinter.messagebox as msgbox
            msgbox.showinfo("GPU Acceleration Info", 
                "GPU Acceleration can speed up:\n" +
                "✅ Training (LightGBM models)\n" +
                "✅ Forecasting (model inference)\n" +
                "❌ Sorting (CPU only)\n" +
                "❌ Processing (CPU only)\n\n" +
                "Requirements:\n" +
                "• NVIDIA GPU with CUDA support\n" +
                "• LightGBM compiled with GPU support\n" +
                "• If disabled, will use CPU (safer)")
        
        ttk.Button(tuning_frame, text="ℹ️", width=3, command=show_gpu_info).grid(row=0, column=2, padx=5, pady=2)
        
        # Hyperparameter tuning option  
        self.batch_tune_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tuning_frame, text="🔍 Enable Optuna Tuning (Training Only)", variable=self.batch_tune_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        
        # Tuning parameters
        ttk.Label(tuning_frame, text="Trials:").grid(row=2, column=0, sticky="w", pady=2)
        self.batch_trials_var = tk.StringVar(value="50")
        trials_entry = ttk.Entry(tuning_frame, textvariable=self.batch_trials_var, width=10)
        trials_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(tuning_frame, text="Timeout (seconds):").grid(row=3, column=0, sticky="w", pady=2)
        self.batch_timeout_var = tk.StringVar(value="300")
        timeout_entry = ttk.Entry(tuning_frame, textvariable=self.batch_timeout_var, width=10)
        timeout_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        
        # File filtering section
        filter_frame = ttk.LabelFrame(batch_frame, text="File Filtering", padding="10")
        filter_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        filter_frame.columnconfigure(1, weight=1)
        
        ttk.Label(filter_frame, text="File Extensions:").grid(row=0, column=0, sticky="w", pady=2)
        self.batch_file_extensions_var = tk.StringVar(value="*.csv,*.parquet")
        file_ext_entry = ttk.Entry(filter_frame, textvariable=self.batch_file_extensions_var, width=40)
        file_ext_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(filter_frame, text="File Pattern:").grid(row=1, column=0, sticky="w", pady=2)
        self.batch_file_pattern_var = tk.StringVar(value="")
        pattern_entry = ttk.Entry(filter_frame, textvariable=self.batch_file_pattern_var, width=40)
        pattern_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Configuration section
        config_frame = ttk.LabelFrame(batch_frame, text="Configuration", padding="10")
        config_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="Config File:").grid(row=0, column=0, sticky="w", pady=2)
        self.batch_config_var = tk.StringVar(value="config/production_config.json")
        config_entry = ttk.Entry(config_frame, textvariable=self.batch_config_var, width=50)
        config_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_batch_config).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(config_frame, text="Processing Mode:").grid(row=1, column=0, sticky="w", pady=2)
        mode_label = ttk.Label(config_frame, text="Sequential (Optimized for Massive Files)", foreground="green")
        mode_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Add info about sequential processing
        def show_sequential_info():
            import tkinter.messagebox as msgbox
            msgbox.showinfo("Sequential Processing Info", 
                "� Optimized for Tens of Millions of Rows:\n\n" +
                "• Sequential processing prevents resource conflicts\n" +
                "• Maximum memory available to each file\n" +
                "• Streaming processor handles massive datasets\n" +
                "• Robust error handling and recovery\n\n" +
                "💡 Benefits for Large Files:\n" +
                "• No memory competition between files\n" +
                "• Full CPU resources for each processing task\n" +
                "• Better stability and reliability\n" +
                "• Easier debugging and progress tracking")
        
        ttk.Button(config_frame, text="ℹ️", width=3, command=show_sequential_info).grid(row=1, column=2, padx=5, pady=2)
        
        # Resource management settings with streaming info
        ttk.Label(config_frame, text="Max Memory per File (GB):").grid(row=2, column=0, sticky="w", pady=2)
        self.batch_max_memory_var = tk.StringVar(value="2.0")
        memory_entry = ttk.Entry(config_frame, textvariable=self.batch_max_memory_var, width=10)
        memory_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Add info button for memory settings
        def show_memory_info():
            import tkinter.messagebox as msgbox
            msgbox.showinfo("Memory Management Info", 
                "🔧 Streaming Processor Benefits:\n\n" +
                "✅ Automatic memory management\n" +
                "✅ Dynamic chunk sizing (10K-1M rows)\n" +
                "✅ Multiple backends (Polars/Vaex/Dask/Pandas)\n" +
                "✅ Out-of-core processing for massive datasets\n" +
                "✅ Intelligent backend selection\n\n" +
                "💾 Memory Settings:\n" +
                "• Files >500K rows use streaming processor\n" +
                "• Chunk size adapts to available memory\n" +
                "• Memory threshold: 75% (adjustable)\n" +
                "• Automatic fallback to legacy if needed")
        
        ttk.Button(config_frame, text="ℹ️", width=3, command=show_memory_info).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(config_frame, text="Processing Timeout (min):").grid(row=3, column=0, sticky="w", pady=2)
        self.batch_timeout_var = tk.StringVar(value="30")
        timeout_entry = ttk.Entry(config_frame, textvariable=self.batch_timeout_var, width=10)
        timeout_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(batch_frame)
        control_frame.grid(row=5, column=0, columnspan=3, pady=(10, 5))
        
        ttk.Button(control_frame, text="🔍 Scan Folder", command=self.scan_batch_folder, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🚀 Start Batch Processing", command=self.start_batch_processing, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏹️ Stop Processing", command=self.stop_batch_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📁 Open Output", command=self.open_batch_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🧹 Clear Log", command=self.clear_batch_log).pack(side=tk.LEFT, padx=5)
        
        # Status section
        status_frame = ttk.LabelFrame(batch_frame, text="Status", padding="5")
        status_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        self.batch_status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.batch_status_var, foreground="blue").grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(status_frame, text="Progress:").grid(row=1, column=0, sticky="w")
        self.batch_progress_var = tk.StringVar(value="0/0 files")
        ttk.Label(status_frame, textvariable=self.batch_progress_var, foreground="green").grid(row=1, column=1, sticky="w", padx=10)
        
        # Log area (reduced height since the whole tab is now scrollable)
        log_frame = ttk.LabelFrame(batch_frame, text="Processing Log", padding="5")
        log_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Reduced height since the whole interface is now scrollable
        self.batch_log_text = scrolledtext.ScrolledText(log_frame, height=10, width=120, wrap=tk.WORD)
        self.batch_log_text.grid(row=0, column=0, sticky="nsew")
        
        # Store canvas reference for potential future use
        self.batch_canvas = canvas
        
        # Initialize batch processing variables
        self.batch_files_list = []
        self.batch_processing_active = False
        self.batch_current_file_index = 0
        
        # Add initial log message
        self.batch_log_text.insert(tk.END, "� MASSIVE SCALE Batch Processing System (Enterprise Ready)\n")
        self.batch_log_text.insert(tk.END, "=" * 80 + "\n\n")
        self.batch_log_text.insert(tk.END, "� BUILT FOR HUNDREDS OF EXTREMELY LARGE FILES:\n")
        self.batch_log_text.insert(tk.END, "• Intelligent parallel processing with automatic resource management\n")
        self.batch_log_text.insert(tk.END, "• Advanced batch strategy analysis (in-memory/streaming/controlled)\n")
        self.batch_log_text.insert(tk.END, "• Multi-threaded workers with optimal batch sizing\n")
        self.batch_log_text.insert(tk.END, "• Memory-aware processing with dynamic scaling\n")
        self.batch_log_text.insert(tk.END, "• Progress persistence and error recovery\n")
        self.batch_log_text.insert(tk.END, "• Emergency fallback for maximum reliability\n\n")
        self.batch_log_text.insert(tk.END, "🔧 PROCESSING STRATEGIES:\n")
        self.batch_log_text.insert(tk.END, "• Small datasets: Parallel in-memory processing\n")
        self.batch_log_text.insert(tk.END, "• Large files (>5GB): Sequential streaming\n")
        self.batch_log_text.insert(tk.END, "• Many files (>100): Controlled parallel batching\n")
        self.batch_log_text.insert(tk.END, "• Mixed loads: Standard parallel with auto-scaling\n\n")
        self.batch_log_text.insert(tk.END, "📋 QUICK START:\n")
        self.batch_log_text.insert(tk.END, "1. Select input folder with your data files\n")
        self.batch_log_text.insert(tk.END, "2. Choose output folder for processed results\n") 
        self.batch_log_text.insert(tk.END, "3. Select processing action (system optimizes automatically):\n")
        self.batch_log_text.insert(tk.END, "   • Data Processing: MASSIVE parallel streaming with feature engineering\n")
        self.batch_log_text.insert(tk.END, "   • Data Sorting: Intelligent stream-based organization by granary\n")
        self.batch_log_text.insert(tk.END, "   • Model Training: Optimized sequential ML training pipeline\n")
        self.batch_log_text.insert(tk.END, "   • Forecasting: Parallel prediction generation\n")
        self.batch_log_text.insert(tk.END, "4. Set memory limits (system auto-detects optimal workers)\n")
        self.batch_log_text.insert(tk.END, "5. Click 'Scan Folder' for intelligent batch analysis\n")
        self.batch_log_text.insert(tk.END, "6. Click 'Start Batch Processing' for enterprise-grade processing\n\n")
        self.batch_log_text.insert(tk.END, "⚡ PERFORMANCE FEATURES:\n")
        self.batch_log_text.insert(tk.END, "• System automatically chooses optimal processing strategy\n")
        self.batch_log_text.insert(tk.END, "• Parallel workers scale based on available resources\n")
        self.batch_log_text.insert(tk.END, "• Memory usage monitored and optimized in real-time\n")
        self.batch_log_text.insert(tk.END, "• Batch sizing adapts to file sizes and system capacity\n")
        self.batch_log_text.insert(tk.END, "• Progress tracking with detailed success/failure reporting\n")
        self.batch_log_text.insert(tk.END, "• Automatic cleanup and resource management\n\n")
        self.batch_log_text.insert(tk.END, "🛡️ RELIABILITY:\n")
        self.batch_log_text.insert(tk.END, "• Skip already processed files (timestamp checking)\n")
        self.batch_log_text.insert(tk.END, "• Continue processing if individual files fail\n")
        self.batch_log_text.insert(tk.END, "• Emergency fallback processing for maximum success rate\n")
        self.batch_log_text.insert(tk.END, "• Comprehensive error logging and recovery\n")
        self.batch_log_text.insert(tk.END, "📦 Ready to handle enterprise-scale data processing workloads!\n\n")
        
        # Initial scroll to show all content is accessible
        self.auto_scroll_batch_log()
        
    def test_connection(self):
        """Test HTTP service connection with modern status updates"""
        try:
            url = self.url_var.get().rstrip('/')
            self.http_status_var.set("🔄 Testing connection...")
            self.connection_status.config(text="🟡 Testing...", fg='#F59E0B')
            self.root.update()
            
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                # Success - update both status indicators
                self.connection_status.config(text="🟢 Connected", fg='#10B981')
                self.http_status_var.set("✅ Service connected successfully - Ready to send requests")
                messagebox.showinfo("Connection Success", 
                                  f"🟢 HTTP service is running and healthy!\n\nURL: {url}\nStatus: Ready for requests")
            else:
                # Partial success
                self.connection_status.config(text="🟡 Limited", fg='#F59E0B') 
                self.http_status_var.set(f"⚠️ Service returned status {response.status_code}")
                messagebox.showerror("Connection Warning", f"Service returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            # Connection failed
            self.connection_status.config(text="🔴 Connection Failed", fg='#EF4444')
            self.http_status_var.set("❌ Connection failed - Check if service is running")
            messagebox.showerror("Connection Error", f"Could not connect to service at {url}\n\nPlease check:\n• Service is running\n• URL is correct\n• No firewall blocking")
        except requests.exceptions.Timeout:
            # Timeout
            self.connection_status.config(text="🔴 Timeout", fg='#EF4444')
            self.http_status_var.set("❌ Connection timeout - Service may be overloaded")
            messagebox.showerror("Connection Timeout", "Connection timed out after 10 seconds")
        except Exception as e:
            # Other errors
            self.connection_status.config(text="🔴 Error", fg='#EF4444')
            self.http_status_var.set(f"❌ Connection error: {str(e)}")
            messagebox.showerror("Connection Error", f"An error occurred: {str(e)}")
            
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
            
            # Update file info display
            try:
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                file_name = os.path.basename(filename)
                file_ext = os.path.splitext(filename)[1].upper()
                self.file_info_var.set(f"📄 {file_name} ({file_ext}, {file_size:.1f} MB)")
                
                # Show detailed file info in response area
                if filename.lower().endswith('.parquet'):
                    self._show_parquet_info(filename, self.http_response_text)
                elif filename.lower().endswith('.csv'):
                    self._show_csv_info(filename, self.http_response_text)
            except Exception as e:
                self.file_info_var.set(f"⚠️ Error reading file: {str(e)}")
        else:
            self.file_info_var.set("No file selected")
    
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
        """Send the selected file to the HTTP service with enhanced feedback"""
        # Get configuration
        service_url = self.url_var.get().rstrip('/')
        endpoint = self.endpoint_var.get()
        file_path = self.file_var.get()

        # Determine if the chosen endpoint needs a file upload
        file_required = endpoint in ["/pipeline", "/process", "/sort"]

        if file_required and not file_path:
            messagebox.showerror("File Required", 
                               f"The {endpoint} endpoint requires a data file.\n\nPlease select a CSV or Parquet file first.")
            return

        if not service_url:
            messagebox.showerror("URL Required", "Please enter a service URL first.")
            return

        # Clear response area with modern styling
        self.http_response_text.delete(1.0, tk.END)
        self.http_response_text.insert(tk.END, f"🚀 Sending Request\n")
        self.http_response_text.insert(tk.END, "=" * 50 + "\n")
        self.http_response_text.insert(tk.END, f"📍 URL: {service_url}{endpoint}\n")
        self.http_response_text.insert(tk.END, f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if file_path:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.http_response_text.insert(tk.END, f"📄 File: {os.path.basename(file_path)} ({file_size:.1f} MB)\n")
        self.http_response_text.insert(tk.END, "🔄 Processing request...\n")
        self.http_response_text.insert(tk.END, "-" * 50 + "\n\n")

        # Update status
        self.http_status_var.set(f"🔄 Sending request to {endpoint}...")
        
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
        """Display CSV response content with enhanced formatting"""
        try:
            # Decode CSV content
            csv_content = response.content.decode('utf-8')
            
            # Enhanced header information
            text_widget.insert(tk.END, f"📊 CSV Response Data\n")
            text_widget.insert(tk.END, "=" * 30 + "\n")
            text_widget.insert(tk.END, f"📏 File size: {len(response.content):,} bytes ({len(response.content)/(1024*1024):.2f} MB)\n")
            
            # Analyze CSV structure
            lines = csv_content.split('\n')
            if lines and lines[0].strip():
                headers = lines[0].split(',')
                data_rows = len([line for line in lines[1:] if line.strip()])
                text_widget.insert(tk.END, f"📋 Columns: {len(headers)}\n")
                text_widget.insert(tk.END, f"📊 Data rows: {data_rows:,}\n")
                text_widget.insert(tk.END, f"🏷️ Headers: {', '.join(headers[:5])}")
                if len(headers) > 5:
                    text_widget.insert(tk.END, f" ... (+{len(headers)-5} more)")
                text_widget.insert(tk.END, "\n")
            
            text_widget.insert(tk.END, "-" * 50 + "\n")
            
            # Show preview of data (first 1000 characters)
            if len(csv_content) > 1000:
                text_widget.insert(tk.END, "📄 Data Preview (first 1000 characters):\n")
                text_widget.insert(tk.END, csv_content[:1000])
                text_widget.insert(tk.END, f"\n\n... ({len(csv_content)-1000:,} more characters)\n")
                text_widget.insert(tk.END, "💡 Full CSV data available - save response if needed\n")
            else:
                text_widget.insert(tk.END, "📄 Complete CSV Data:\n")
                text_widget.insert(tk.END, csv_content)
            
            text_widget.insert(tk.END, "\n" + "-" * 50 + "\n")
            
        except Exception as e:
            text_widget.insert(tk.END, f"❌ Error displaying CSV response: {str(e)}\n")
            text_widget.insert(tk.END, f"Raw response length: {len(response.content)} bytes\n")
            
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
                if (line.startswith('ID') or line.startswith('-') or 
                    line.startswith('Available') or line.startswith('=') or
                    'granary_id' in line.lower() or 'granary_name' in line.lower()):
                    continue
                if len(line.strip()) > 0 and not line.startswith('#'):
                    parts = [part.strip() for part in line.split() if part.strip()]
                    if len(parts) >= 3:
                        granary_id = parts[0]
                        granary_name = None
                        for part in parts[1:]:
                            if (len(part) > 2 and 
                                (any('\u4e00' <= char <= '\u9fff' for char in part) or 
                                 len(part) > 10)):
                                granary_name = part
                                break
                        if granary_name and granary_name != 'Name':
                            granary_data.append((granary_name, granary_id))
                elif 'Granary:' in line:
                    parts = line.split('Granary:')
                    if len(parts) > 1:
                        granary_part = parts[1].split('(ID:')[0].strip()
                        if granary_part:
                            id_match = re.search(r'ID:\s*([^\s,]+)', line)
                            granary_id = id_match.group(1) if id_match else granary_part
                            granary_data.append((granary_part, granary_id))
            self.granaries_data = granary_data
            granary_names = [item[0] for item in granary_data]
            self.granary_combo['values'] = granary_names
            if granary_names:
                self.granary_combo.set(granary_names[0])
        except Exception as e:
            self.explorer_response_text.insert(tk.END, f"\nError parsing granaries: {str(e)}\n")
            self.explorer_response_text.insert(tk.END, f"Raw output lines: {output_lines[:5]}\n")

    def get_silos_for_granary(self):
        """Get silos for the selected granary"""
        selected_granary_name = self.granary_selection_var.get()
        if not selected_granary_name:
            messagebox.showerror("Error", "Please select a granary first")
            return
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
        self.explorer_response_text.delete(1.0, tk.END)
        self.explorer_response_text.insert(tk.END, f"📦 Getting Silos for Granary: {selected_granary_name} (ID: {selected_granary_id})\n")
        self.explorer_response_text.insert(tk.END, f"Config: {config_file}\n")
        self.explorer_response_text.insert(tk.END, "-" * 60 + "\n\n")
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
            for line in output_lines:
                self.root.after(0, self.explorer_response_text.insert, tk.END, line)
                self.root.after(0, self.explorer_response_text.see, tk.END)
            self.root.after(0, self.explorer_status_var.set, "")
            if success:
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
                if 'Silo:' in line:
                    parts = line.split('Silo:')
                    if len(parts) > 1:
                        silo_part = parts[1].split('(ID:')[0].strip()
                        if silo_part and silo_part not in silo_names:
                            silo_names.append(silo_part)
                elif 'silo_name' in line.lower() or 'store_name' in line.lower():
                    if '|' in line and not line.startswith('-') and not line.startswith('ID'):
                        parts = line.split('|')
                        if len(parts) >= 2:
                            silo_name = parts[1].strip()
                            if silo_name and silo_name != 'silo_name' and silo_name != 'store_name':
                                silo_names.append(silo_name)
            self.silos_data = silo_names
            self.silo_combo['values'] = silo_names
            if silo_names:
                self.silo_combo.set(silo_names[0])
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
            # GET endpoints that don't require files
            if endpoint in ["/health", "/models", "/forecast"]:
                response = requests.get(f"{remote_url}{endpoint}", timeout=30)
            # POST endpoints - test without file to see if endpoint exists
            elif endpoint in ["/sort", "/process", "/pipeline"]:
                response = requests.post(f"{remote_url}{endpoint}", timeout=30)
            # Train endpoint (POST)
            elif endpoint == "/train":
                response = requests.post(f"{remote_url}{endpoint}", timeout=30)
            else:
                response = requests.get(f"{remote_url}{endpoint}", timeout=30)
            
            self.root.after(0, self.remote_response_text.insert, tk.END, f"Status Code: {response.status_code}\n")
            
            # Handle different success codes for different endpoint types
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Success!\n")
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {json.dumps(json_response, indent=2)}\n")
                except:
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Success!\n")
                    self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {response.text}\n")
                self.root.after(0, self.remote_status_var.set, f"{endpoint} test successful")
            elif response.status_code in [400, 422] and endpoint in ["/sort", "/process", "/pipeline", "/train"]:
                # POST endpoints without required files return 400/422, which means endpoint is reachable
                self.root.after(0, self.remote_response_text.insert, tk.END, f"✅ Endpoint reachable (missing file is expected)\n")
                self.root.after(0, self.remote_response_text.insert, tk.END, f"Response: {response.text}\n")
                self.root.after(0, self.remote_status_var.set, f"{endpoint} endpoint reachable")
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
        endpoints = ["/health", "/models", "/sort", "/process", "/train", "/forecast", "/pipeline"]
        results = {}
        
        for endpoint in endpoints:
            self.root.after(0, self.remote_response_text.insert, tk.END, f"Testing {endpoint}...\n")
            
            try:
                # GET endpoints that don't require files
                if endpoint in ["/health", "/models", "/forecast"]:
                    response = requests.get(f"{remote_url}{endpoint}", timeout=30)
                # POST endpoints that require files - we can't test these without data
                elif endpoint in ["/sort", "/process", "/pipeline"]:
                    # For POST endpoints, we just check if the endpoint exists and returns proper error
                    response = requests.post(f"{remote_url}{endpoint}", timeout=30)
                    # Accept 400 (Bad Request) as a valid response for POST endpoints without files
                    if response.status_code in [400, 422]:  # 422 for validation errors
                        results[endpoint] = "✅ PASS (endpoint reachable)"
                        self.root.after(0, self.remote_response_text.insert, tk.END, f"  {endpoint}: ✅ PASS (endpoint reachable)\n")
                        continue
                # GET endpoint for train (check if it accepts GET)
                elif endpoint == "/train":
                    response = requests.post(f"{remote_url}{endpoint}", timeout=30)
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

    # ------------------------------------------------------------------
    # PRODUCTION PIPELINE METHODS
    # ------------------------------------------------------------------

    def check_system_resources(self):
        """Check current system resource usage"""
        try:
            import psutil
            
            # Get system stats
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            stats_text = (
                f"💾 Memory: {memory.percent:.1f}% used ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB) | "
                f"🔥 CPU: {cpu_percent:.1f}% | "
                f"💽 Disk: {disk.percent:.1f}% used"
            )
            
            self.system_stats_var.set(stats_text)
            
            # Log to output
            self.retrieval_response_text.insert(tk.END, f"\n📊 System Resources Check:\n{stats_text}\n")
            self.retrieval_response_text.see(tk.END)
            
        except ImportError:
            self.system_stats_var.set("⚠️ psutil not available - install with: pip install psutil")
        except Exception as e:
            self.system_stats_var.set(f"❌ Error checking system: {str(e)}")

    def open_task_manager(self):
        """Open Windows Task Manager"""
        try:
            subprocess.Popen('taskmgr')
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Task Manager: {str(e)}")

    def run_production_pipeline(self):
        """Run the full production pipeline using subprocess approach"""
        try:
            self.retrieval_status_var.set("Running Production Pipeline...")
            config_path = self.retrieval_cfg_var.get()
            
            self.retrieval_response_text.insert(tk.END, f"\n🚀 Starting Production Pipeline...\n")
            self.retrieval_response_text.insert(tk.END, f"Config: {config_path}\n")
            self.retrieval_response_text.insert(tk.END, f"Retrieval: {'✓' if self.run_retrieval_var.get() else '✗'} | ")
            self.retrieval_response_text.insert(tk.END, f"Preprocessing: {'✓' if self.run_preprocessing_var.get() else '✗'} | ")
            self.retrieval_response_text.insert(tk.END, f"Training: {'✓' if self.run_training_var.get() else '✗'}\n")
            self.retrieval_response_text.insert(tk.END, "="*60 + "\n")
            self.root.update()
            
            # Build command to run production pipeline as subprocess
            pipeline_script = Path(__file__).parent.parent.parent / "production_pipeline.py"
            
            # Build command arguments
            cmd = [self.get_python_executable(), str(pipeline_script)]
            cmd.extend(["--config", config_path])
            
            # Add phase control arguments (skip what's not selected)
            if not self.run_retrieval_var.get():
                cmd.append("--skip-retrieval")
            if not self.run_preprocessing_var.get():
                cmd.append("--skip-preprocessing")
            if not self.run_training_var.get():
                cmd.append("--skip-training")
            
            # Add granary filter if specified
            if hasattr(self, 'retrieval_granary_var') and self.retrieval_granary_var.get():
                cmd.extend(["--granary", self.retrieval_granary_var.get()])
            
            # Add max records if specified for testing
            if hasattr(self, 'max_records_var') and self.max_records_var.get():
                cmd.extend(["--max-records", self.max_records_var.get()])
            
            # Add mode-specific arguments
            if hasattr(self, 'retrieval_mode_var'):
                mode = self.retrieval_mode_var.get()
                
                if mode == "full-retrieval":
                    cmd.append("--full-retrieval")
                elif mode == "incremental":
                    cmd.append("--incremental")
                    if hasattr(self, 'days_var') and self.days_var.get():
                        cmd.extend(["--days", self.days_var.get()])
                elif mode == "date-range":
                    cmd.append("--date-range")
                    if hasattr(self, 'start_date_var') and self.start_date_var.get():
                        cmd.extend(["--start", self.start_date_var.get()])
                    if hasattr(self, 'end_date_var') and self.end_date_var.get():
                        cmd.extend(["--end", self.end_date_var.get()])
                else:
                    # Default to full-retrieval if no mode specified
                    cmd.append("--full-retrieval")
            
            self.retrieval_response_text.insert(tk.END, f"Command: {' '.join(cmd)}\n\n")
            self.root.update()
            
            # Run in thread to avoid blocking GUI
            thread = threading.Thread(target=self._run_pipeline_subprocess, args=(cmd,))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            error_msg = f"Production Pipeline Error: {str(e)}"
            self.retrieval_response_text.insert(tk.END, f"\n❌ {error_msg}\n")
            self.retrieval_status_var.set("Pipeline Failed")
            messagebox.showerror("Pipeline Error", error_msg)

    def _run_pipeline_subprocess(self, cmd):
        """Run the production pipeline as a subprocess"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            if process.stdout is None:
                self.root.after(0, self._show_pipeline_error, "Pipeline Error", "No output captured from subprocess")
                return
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Update GUI with the line
                    self.root.after(0, self.retrieval_response_text.insert, tk.END, line)
                    self.root.after(0, self.retrieval_response_text.see, tk.END)
                    
                    # Update progress for key messages
                    if any(keyword in line.lower() for keyword in ["starting", "processing", "training", "completed"]):
                        self.root.after(0, self.retrieval_status_var.set, line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                self.root.after(0, self.retrieval_status_var.set, "Production Pipeline completed successfully")
                self.root.after(0, self.retrieval_response_text.insert, tk.END, "\n✅ Production Pipeline completed successfully!\n")
            else:
                self.root.after(0, self.retrieval_status_var.set, f"Production Pipeline failed (exit code {process.returncode})")
                self.root.after(0, self.retrieval_response_text.insert, tk.END, f"\n❌ Production Pipeline failed with exit code {process.returncode}\n")
                
        except Exception as exc:
            self.root.after(0, self._show_pipeline_error, "Pipeline Error", str(exc))

    def _show_pipeline_error(self, title, message):
        """Show error message for pipeline operations"""
        messagebox.showerror(title, message)
        self.retrieval_status_var.set("Pipeline Error")

    def run_quick_test(self):
        """Run a quick test with limited records"""
        # Set max records to 10K for quick test
        self.max_records_var.set("10000")
        self.retrieval_response_text.insert(tk.END, f"\n⚡ Quick Test Mode: Limited to 10,000 records per granary\n")
        self.run_production_pipeline()

    def check_pipeline_status(self):
        """Check the status of running pipeline processes"""
        try:
            import psutil
            
            self.retrieval_response_text.insert(tk.END, f"\n🔄 Checking Pipeline Status...\n")
            
            # Look for Python processes running pipeline-related scripts
            pipeline_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                        cmdline = ' '.join(proc.info['cmdline'])
                        if any(keyword in cmdline.lower() for keyword in 
                               ['production_pipeline', 'automated_data_retrieval', 'granary_pipeline']):
                            pipeline_processes.append({
                                'pid': proc.info['pid'],
                                'cmdline': cmdline,
                                'memory_mb': proc.memory_info().rss // (1024*1024),
                                'cpu_percent': proc.cpu_percent()
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if pipeline_processes:
                self.retrieval_response_text.insert(tk.END, f"🔍 Found {len(pipeline_processes)} pipeline process(es):\n")
                for proc in pipeline_processes:
                    self.retrieval_response_text.insert(tk.END, 
                        f"  PID {proc['pid']}: {proc['memory_mb']:.0f}MB, {proc['cpu_percent']:.1f}% CPU\n")
            else:
                self.retrieval_response_text.insert(tk.END, "✅ No active pipeline processes found\n")
            
            # Check system resources
            self.check_system_resources()
            
        except ImportError:
            self.retrieval_response_text.insert(tk.END, "⚠️ psutil not available for status checking\n")
        except Exception as e:
            self.retrieval_response_text.insert(tk.END, f"❌ Status check error: {str(e)}\n")

    # Simple Data Retrieval Methods
    def browse_simple_output_dir(self):
        """Browse for simple retrieval output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.simple_output_var.set(directory)
    
    def list_all_silos(self):
        """List all available silos from the database"""
        def run_list():
            try:
                self.simple_progress_var.set("Listing all silos...")
                self.simple_response_text.insert(tk.END, "\n🔍 Retrieving list of all silos...\n")
                
                # Use the simple data retrieval script to get silo list
                script_path = Path(__file__).parent.parent / "simple_data_retrieval.py"
                config_path = Path(__file__).parent.parent.parent / "config" / "production_config.json"
                
                # Create a temporary script to list silos
                temp_script = f'''
import sys
sys.path.insert(0, r"{Path(__file__).parent.parent.parent}")
from scripts.simple_data_retrieval import SimpleDataRetriever, load_config

config = load_config(r"{config_path}")
retriever = SimpleDataRetriever(config["database"])
df = retriever.get_all_granaries_and_silos()
print("\\n=== All Available Silos ===")
for _, row in df.iterrows():
    print(f"Granary: {{row['storepoint_id']}} | Silo: {{row['store_name']}} ({{row['store_id']}}) | Sub-table: {{row['sub_table_id']}}")
'''
                
                # Execute the temporary script
                result = subprocess.run([
                    self.get_python_executable(), "-c", temp_script
                ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
                
                self.root.after(0, self.simple_response_text.insert, tk.END, result.stdout + "\n")
                if result.stderr:
                    self.root.after(0, self.simple_response_text.insert, tk.END, f"Errors: {result.stderr}\n")
                
                self.root.after(0, self.simple_progress_var.set, "Silo listing completed")
                
            except Exception as e:
                self.root.after(0, self.simple_response_text.insert, tk.END, f"\n❌ Error listing silos: {e}\n")
                self.root.after(0, self.simple_progress_var.set, "Error listing silos")
        
        threading.Thread(target=run_list, daemon=True).start()
    
    def get_granaries_and_silos(self):
        """Get all granaries and their corresponding silos using the user's query"""
        def run_query():
            try:
                self.simple_progress_var.set("Retrieving granaries and silos...")
                self.simple_response_text.insert(tk.END, "\n🏢 Retrieving all granaries and their silos...\n")
                
                # Path to the simple data retrieval script
                config_path = Path(__file__).parent.parent.parent / "config" / "production_config.json"
                
                # Create a temporary script to run the user's query
                temp_script = f'''
import sys
sys.path.insert(0, r"{Path(__file__).parent.parent.parent}")
from scripts.simple_data_retrieval import SimpleDataRetriever, load_config

config = load_config(r"{config_path}")
retriever = SimpleDataRetriever(config["database"])

# Run the user's specific query
query = """
    SELECT 
        loc.storepoint_id,
        locs.store_name as granary_name, 
        loc.sub_table_id,
        store.store_id, 
        store.store_name as silo_name 
    FROM cloud_server.base_store_location_other loc
    INNER JOIN cloud_server.v_store_list locs 
        ON locs.store_id = loc.storepoint_id 
        AND locs.level = '1'
    INNER JOIN cloud_server.v_store_list store 
        ON store.storepoint_id = loc.storepoint_id 
        AND store.level = '4'
    ORDER BY loc.sub_table_id, store.store_name
"""

import pandas as pd
df = pd.read_sql(query, retriever.engine)

print("\\n=== Granaries and Their Silos with Date Ranges ===")
print(f"Found {{len(df)}} silos to process")
print("\\nProcessing silos for date ranges (this may take a moment)...")
print()

# Prepare data for CSV with date ranges
csv_data = []
processed_count = 0

# Group by granary for better display
grouped = df.groupby(['storepoint_id', 'granary_name', 'sub_table_id'])
total_granaries = len(grouped)
granary_count = 0

for (granary_id, granary_name, sub_table_id), group in grouped:
    granary_count += 1
    print(f"[GRANARY {{granary_count}}/{{total_granaries}}] {{granary_name}} (ID: {{granary_id}}, Sub-table: {{sub_table_id}})")
    sys.stdout.flush()  # Force output immediately
    
    silo_count = 0
    total_silos_in_granary = len(group)
    
    for _, silo in group.iterrows():
        silo_count += 1
        processed_count += 1
        
        try:
            print(f"   [{{silo_count}}/{{total_silos_in_granary}}] Processing {{silo['silo_name']}} ({{silo['store_id']}}...)...", end=' ')
            sys.stdout.flush()
            
            # Get date range for this silo
            min_date, max_date = retriever.get_silo_date_range(silo['store_id'], sub_table_id)
            
            if min_date and max_date:
                date_info = f"{{min_date.strftime('%Y-%m-%d')}} to {{max_date.strftime('%Y-%m-%d')}}"
                days_span = (max_date - min_date).days
                print(f"✅ {{date_info}} ({{days_span}} days)")
                data_available = 'Yes'
            else:
                date_info = "No data available"
                min_date = max_date = None
                days_span = 0
                data_available = 'No'
                print("❌ No data")
            
            # Add to CSV data
            csv_data.append({{
                'granary_id': granary_id,
                'granary_name': granary_name,
                'sub_table_id': sub_table_id,
                'silo_id': silo['store_id'],
                'silo_name': silo['silo_name'],
                'start_date': min_date.strftime('%Y-%m-%d') if min_date else None,
                'end_date': max_date.strftime('%Y-%m-%d') if max_date else None,
                'days_span': days_span,
                'data_available': data_available
            }})
            
        except Exception as e:
            print(f"⚠️ Error: {{str(e)[:50]}}")
            csv_data.append({{
                'granary_id': granary_id,
                'granary_name': granary_name,
                'sub_table_id': sub_table_id,
                'silo_id': silo['store_id'],
                'silo_name': silo['silo_name'],
                'start_date': None,
                'end_date': None,
                'days_span': 0,
                'data_available': 'Error'
            }})
        
        sys.stdout.flush()  # Ensure immediate output
    print()

# Save detailed CSV with date ranges
output_file = "data/simple_retrieval/granaries_silos_with_dates.csv"
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[SAVED] Detailed data with date ranges saved to: {{output_file}}")

# Show summary statistics
silos_with_data = csv_df[csv_df['data_available'] == 'Yes']
print("\\n[SUMMARY]")
print(f"   Processed silos: {{len(csv_df)}}")
print(f"   Granaries: {{csv_df['granary_id'].nunique()}}")
print(f"   Silos with data: {{len(silos_with_data)}}")
print(f"   Silos without data: {{len(csv_df) - len(silos_with_data)}}")
print(f"   Sub-tables used: {{sorted(csv_df['sub_table_id'].unique())}}")

if len(silos_with_data) > 0:
    print(f"   Average data span: {{silos_with_data['days_span'].mean():.0f}} days")
    print(f"   Longest data span: {{silos_with_data['days_span'].max()}} days")
    print(f"   Shortest data span: {{silos_with_data['days_span'].min()}} days")

# Show top granaries with data
data_counts = silos_with_data.groupby('granary_name').size().sort_values(ascending=False)
print("\\n[TOP GRANARIES] Granaries with most data-enabled silos:")
for granary, count in data_counts.head(5).items():
    total_count = len(csv_df[csv_df['granary_name'] == granary])
    print(f"   {{granary}}: {{count}}/{{total_count}} silos with data")

print("\\n=== Processing Complete ===")
'''
                
                # Execute the temporary script
                import os
                
                # Set environment for real-time output with proper encoding
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONUNBUFFERED'] = '1'
                env['LANG'] = 'en_US.UTF-8'
                env['LC_ALL'] = 'en_US.UTF-8'
                
                try:
                    process = subprocess.Popen([
                        self.get_python_executable(), "-c", temp_script
                    ], 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,  # Unbuffered
                    encoding='utf-8',
                    errors='replace',  # Replace problematic characters
                    cwd=Path(__file__).parent.parent.parent,
                    env=env)
                    
                    # Read output in real-time
                    if process.stdout:
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                self.root.after(0, self.simple_response_text.insert, tk.END, output)
                                self.root.after(0, self.simple_response_text.see, tk.END)
                                self.root.after(0, self.root.update_idletasks)
                    
                    # Also read stderr
                    if process.stderr:
                        stderr_output = process.stderr.read()
                        if stderr_output:
                            self.root.after(0, self.simple_response_text.insert, tk.END, f"\\nErrors: {stderr_output}\\n")
                    
                    return_code = process.poll()
                    
                    if return_code == 0:
                        self.root.after(0, self.simple_progress_var.set, "Granaries and silos retrieval completed")
                    else:
                        self.root.after(0, self.simple_progress_var.set, "Error retrieving granaries and silos")
                        
                except Exception as proc_error:
                    self.root.after(0, self.simple_response_text.insert, tk.END, f"\\n❌ Process error: {proc_error}\\n")
                    self.root.after(0, self.simple_progress_var.set, "Error retrieving granaries and silos")
                
            except Exception as e:
                self.root.after(0, self.simple_response_text.insert, tk.END, f"\n❌ Error retrieving granaries and silos: {e}\n")
                self.root.after(0, self.simple_progress_var.set, "Error retrieving granaries and silos")
        
        threading.Thread(target=run_query, daemon=True).start()
    
    def auto_fill_next_silo(self):
        """Auto-fill the next silo configuration from the stored data"""
        def load_and_fill():
            try:
                self.simple_progress_var.set("Loading silo data...")
                
                # If we don't have silo data loaded, load it first
                if not self.all_silos_data:
                    self.simple_response_text.insert(tk.END, "\n🔄 Loading all silos data first...\n")
                    
                    # Path to config
                    config_path = Path(__file__).parent.parent.parent / "config" / "production_config.json"
                    
                    # Create a script to load all silo data
                    temp_script = f'''
import sys
sys.path.insert(0, r"{Path(__file__).parent.parent.parent}")
from scripts.simple_data_retrieval import SimpleDataRetriever, load_config
import pandas as pd
import json

config = load_config(r"{config_path}")
retriever = SimpleDataRetriever(config["database"])

# Get all granaries and silos
df = retriever.get_granaries_with_details()

# Prepare silo data with date ranges (prioritize silos with data)
silo_data = []
silos_with_data = []
silos_without_data = []

for _, row in df.iterrows():
    try:
        min_date, max_date = retriever.get_silo_date_range(row['store_id'], row['sub_table_id'])
        silo_info = {{
            'granary_name': row['granary_name'],
            'granary_id': row['storepoint_id'],
            'sub_table_id': row['sub_table_id'],
            'silo_id': row['store_id'],
            'silo_name': row['silo_name'],
            'start_date': min_date.strftime('%Y-%m-%d') if min_date else '2024-01-01',
            'end_date': max_date.strftime('%Y-%m-%d') if max_date else '2024-12-31',
            'has_data': bool(min_date and max_date)
        }}
        
        # Separate silos with data from those without
        if silo_info['has_data']:
            silos_with_data.append(silo_info)
        else:
            silos_without_data.append(silo_info)
            
        # Limit to 50 silos with data for reasonable performance
        if len(silos_with_data) >= 50:
            break
            
    except:
        silo_info = {{
            'granary_name': row['granary_name'],
            'granary_id': row['storepoint_id'],
            'sub_table_id': row['sub_table_id'],
            'silo_id': row['store_id'],
            'silo_name': row['silo_name'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'has_data': False
        }}
        silos_without_data.append(silo_info)

# Prioritize silos with data, then add some without data for completeness
silo_data = silos_with_data + silos_without_data[:10]  # Add up to 10 silos without data

print(json.dumps(silo_data, ensure_ascii=False, indent=2))
'''
                    
                    # Execute and get silo data
                    result = subprocess.run([
                        self.get_python_executable(), "-c", temp_script
                    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
                    
                    if result.returncode == 0:
                        try:
                            import json
                            self.all_silos_data = json.loads(result.stdout)
                            # Count silos with data
                            silos_with_data_count = sum(1 for silo in self.all_silos_data if silo.get('has_data', False))
                            self.root.after(0, self.simple_response_text.insert, tk.END, 
                                f"Loaded {len(self.all_silos_data)} silos ({silos_with_data_count} with data) for auto-fill\n")
                        except json.JSONDecodeError:
                            self.root.after(0, self.simple_response_text.insert, tk.END, "Error parsing silo data\n")
                            return
                    else:
                        self.root.after(0, self.simple_response_text.insert, tk.END, f"Error loading silo data: {{result.stderr}}\\n")
                        return
                
                # Fill in the next silo that has data
                if self.all_silos_data:
                    # Find the next silo with data
                    attempts = 0
                    max_attempts = len(self.all_silos_data)
                    
                    while attempts < max_attempts:
                        if self.current_silo_index >= len(self.all_silos_data):
                            self.current_silo_index = 0
                        
                        silo = self.all_silos_data[self.current_silo_index]
                        
                        # Check if this silo has data
                        if silo.get('has_data', False):
                            # Update the GUI fields
                            self.root.after(0, self.simple_granary_var.set, silo['granary_name'])
                            self.root.after(0, self.simple_silo_var.set, silo['silo_id'])
                            self.root.after(0, self.simple_start_date_var.set, silo['start_date'])
                            self.root.after(0, self.simple_end_date_var.set, silo['end_date'])
                            
                            # Show info about the selected silo
                            info_msg = f"\n🔄 Auto-filled silo with data ({self.current_silo_index + 1}/{len(self.all_silos_data)}):\n"
                            info_msg += f"   Granary: {silo['granary_name']}\n"
                            info_msg += f"   Silo: {silo['silo_name']} ({silo['silo_id']})\n"
                            info_msg += f"   Date Range: {silo['start_date']} to {silo['end_date']}\n"
                            info_msg += f"   ✅ Data Available: {silo['has_data']}\n"
                            
                            self.root.after(0, self.simple_response_text.insert, tk.END, info_msg)
                            
                            # Move to next silo for next time
                            self.current_silo_index += 1
                            
                            self.root.after(0, self.simple_progress_var.set, f"Auto-filled silo with data ({self.current_silo_index}/{len(self.all_silos_data)})")
                            break
                        else:
                            # Skip silos without data
                            attempts += 1
                            self.current_silo_index += 1
                            continue
                    
                    if attempts >= max_attempts:
                        self.root.after(0, self.simple_response_text.insert, tk.END, "\n⚠️ No silos with data found in the loaded set. Try 'Get Granaries & Silos' to load more.\n")
                        self.root.after(0, self.simple_progress_var.set, "No silos with data found")
                
                else:
                    self.root.after(0, self.simple_response_text.insert, tk.END, "\\n⚠️ No silo data available for auto-fill. Try 'Get Granaries & Silos' first.\\n")
                    self.root.after(0, self.simple_progress_var.set, "No silo data available")
                
            except Exception as e:
                self.root.after(0, self.simple_response_text.insert, tk.END, f"\\n❌ Error in auto-fill: {{e}}\\n")
                self.root.after(0, self.simple_progress_var.set, "Error in auto-fill")
        
        threading.Thread(target=load_and_fill, daemon=True).start()
    
    def auto_process_all_silos(self):
        """Auto process all silos with data - automatically uses granaries_silos_with_dates.csv if available"""
        # Check if granaries_silos_with_dates.csv exists in the expected locations
        csv_file_found = None
        
        # Get the absolute path to this script file and work backwards to find service directory
        script_path = Path(__file__).resolve()
        
        # Find service directory by walking up the path
        current = script_path.parent
        service_dir = None
        while current.parent != current:  # Not at root
            if current.name == "service":
                service_dir = current
                break
            current = current.parent
        
        if not service_dir:
            # Fallback - assume we're in service/scripts/testing/
            service_dir = script_path.parent.parent.parent
        
        search_paths = [
            service_dir / "data" / "simple_retrieval" / "granaries_silos_with_dates.csv",  # Primary location
            Path("service/data/simple_retrieval/granaries_silos_with_dates.csv"),
            Path("../service/data/simple_retrieval/granaries_silos_with_dates.csv"), 
            Path("data/simple_retrieval/granaries_silos_with_dates.csv"),
            Path("granaries_silos_with_dates.csv")
        ]
        
        print(f"DEBUG: Service directory: {service_dir}")
        print(f"DEBUG: Looking for granaries_silos_with_dates.csv in:")
        for csv_path in search_paths:
            abs_path = csv_path.resolve() if csv_path.is_absolute() else csv_path.absolute()
            exists = csv_path.exists()
            print(f"  {csv_path} -> {abs_path} (exists: {exists})")
            if exists:
                csv_file_found = str(csv_path)
                print(f"  ✅ FOUND: {csv_file_found}")
                break
        
        # Ask user for data source preference
        from tkinter import messagebox, filedialog
        
        if csv_file_found:
            choice = messagebox.askyesnocancel(
                "Data Source Selection",
                f"Found existing CSV file: {Path(csv_file_found).name}\n" +
                f"Location: {csv_file_found}\n\n" +
                "• Yes: Use this CSV file automatically\n" +
                "• No: Use cached data (requires running 'Get Granaries & Silos' first)\n" +
                "• Cancel: Abort operation"
            )
        else:
            choice = messagebox.askyesnocancel(
                "Select Data Source",
                "granaries_silos_with_dates.csv not found in expected locations.\n\n" +
                "• Yes: Browse for CSV file manually\n" +
                "• No: Use cached data (requires running 'Get Granaries & Silos' first)\n" +
                "• Cancel: Abort operation"
            )
        
        if choice is None:  # Cancel
            return
        
        def process_all():
            try:
                self.simple_progress_var.set("Starting auto processing of all silos...")
                self.simple_response_text.insert(tk.END, "\n🚀 Starting Auto Process All Silos...\n")
                
                silos_with_data = []
                
                if choice:  # Yes - Use CSV file
                    if csv_file_found:
                        # Use the automatically found CSV file
                        csv_file = csv_file_found
                        self.root.after(0, self.simple_response_text.insert, tk.END, 
                            f"📂 Using found CSV file: {Path(csv_file).name}\n")
                    else:
                        # Browse for CSV file manually
                        self.root.after(0, self.simple_response_text.insert, tk.END, "📂 Browsing for CSV file...\n")
                        
                        # File selection dialog - use correct simple_retrieval path based on service directory
                        script_path = Path(__file__).resolve()
                        current = script_path.parent
                        while current.parent != current:  # Not at root
                            if current.name == "service":
                                service_dir = current
                                break
                            current = current.parent
                        else:
                            # Fallback
                            service_dir = script_path.parent.parent.parent
                        
                        simple_retrieval_paths = [
                            service_dir / "data" / "simple_retrieval",
                            Path("service/data/simple_retrieval"),
                            Path("../service/data/simple_retrieval"), 
                            Path("data/simple_retrieval"),
                            Path(".")
                        ]
                        
                        initial_dir = "."
                        for path in simple_retrieval_paths:
                            if path.exists():
                                initial_dir = str(path)
                                break
                        
                        csv_file = filedialog.askopenfilename(
                            title="Select Granaries and Silos CSV File",
                            filetypes=[
                                ("CSV files", "*.csv"),
                                ("All files", "*.*")
                            ],
                            initialdir=initial_dir
                        )
                        
                        if not csv_file:
                            self.root.after(0, self.simple_response_text.insert, tk.END, "❌ No file selected. Operation cancelled.\n")
                            self.root.after(0, self.simple_progress_var.set, "Cancelled")
                            return
                    
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_file)
                        
                        # Validate CSV structure
                        required_columns = ['granary_name', 'silo_id', 'silo_name', 'start_date', 'end_date', 'data_available']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        
                        if missing_columns:
                            self.root.after(0, self.simple_response_text.insert, tk.END, 
                                f"❌ CSV file missing required columns: {', '.join(missing_columns)}\n")
                            self.root.after(0, self.simple_progress_var.set, "Invalid CSV format")
                            return
                        
                        # Filter for silos with data
                        data_silos = df[df['data_available'] == 'Yes'].copy()
                        
                        for _, row in data_silos.iterrows():
                            silos_with_data.append({
                                'granary_name': str(row['granary_name']),
                                'silo_id': str(row['silo_id']),
                                'silo_name': str(row['silo_name']),
                                'start_date': str(row['start_date']) if pd.notna(row['start_date']) else '',
                                'end_date': str(row['end_date']) if pd.notna(row['end_date']) else '',
                                'has_data': True
                            })
                        
                        self.root.after(0, self.simple_response_text.insert, tk.END, 
                            f"📊 Loaded {len(silos_with_data)} silos with data from CSV file: {Path(csv_file).name}\n\n")
                        
                    except Exception as e:
                        self.root.after(0, self.simple_response_text.insert, tk.END, f"❌ Error reading CSV file: {e}\n")
                        self.root.after(0, self.simple_progress_var.set, "CSV read error")
                        return
                
                else:  # No - Use cached data
                    self.root.after(0, self.simple_response_text.insert, tk.END, "🔄 Using cached silo data...\n")
                    
                    # Check if we have silo data loaded
                    if not self.all_silos_data:
                        self.root.after(0, self.simple_response_text.insert, tk.END, "🔄 Loading silo data first...\n")
                        # Trigger loading of silo data by calling the auto_fill method first
                        # This will populate self.all_silos_data
                        self.auto_fill_next_silo()
                        
                        # Wait a moment for the data to load
                        import time
                        time.sleep(3)
                        
                        if not self.all_silos_data:
                            self.root.after(0, self.simple_response_text.insert, tk.END, "❌ Failed to load silo data. Try 'Get Granaries & Silos' first.\n")
                            self.root.after(0, self.simple_progress_var.set, "Failed - no silo data")
                            return
                    
                    # Get silos with data from cached data
                    silos_with_data = [silo for silo in self.all_silos_data if silo.get('has_data', False)]
                
                if not silos_with_data:
                    self.root.after(0, self.simple_response_text.insert, tk.END, "⚠️ No silos with data found to process.\n")
                    self.root.after(0, self.simple_progress_var.set, "No silos with data")
                    return
                
                # Filter out silos that already have files in simple_retrieval directory
                self.root.after(0, self.simple_response_text.insert, tk.END, "🔍 Checking for existing files in simple_retrieval directory...\n")
                
                # Import the centralized filtering utility with proper path handling
                try:
                    import sys
                    import os
                    
                    # Find the service directory properly
                    script_path = Path(__file__).resolve()
                    current = script_path.parent
                    while current.parent != current:  # Not at root
                        if current.name == "service":
                            service_dir = current
                            break
                        current = current.parent
                    else:
                        # Fallback
                        service_dir = script_path.parent.parent.parent
                    
                    utils_path = str(service_dir)
                    if utils_path not in sys.path:
                        sys.path.insert(0, utils_path)
                    
                    print(f"DEBUG: Added to sys.path: {utils_path}")
                    
                    from utils.silo_filtering import filter_silos_by_existing_files
                    
                    filtered_silos, skipped_silos = filter_silos_by_existing_files(silos_with_data)
                    
                    self.root.after(0, self.simple_response_text.insert, tk.END, 
                        f"   Skipping {len(skipped_silos)} silos that already have data files\n")
                    
                    # Update silos_with_data to the filtered list
                    silos_with_data = filtered_silos
                    
                except ImportError as e:
                    self.root.after(0, self.simple_response_text.insert, tk.END, 
                        f"   Warning: Could not load filtering utility ({e}) - processing all silos\n")
                    print(f"DEBUG: Import error: {e}")
                except Exception as e:
                    self.root.after(0, self.simple_response_text.insert, tk.END, 
                        f"   Warning: Error in filtering ({e}) - processing all silos\n")
                    print(f"DEBUG: Filtering error: {e}")
                
                if not silos_with_data:
                    self.root.after(0, self.simple_response_text.insert, tk.END, "✅ All silos already have data files - nothing to process!\n")
                    self.root.after(0, self.simple_progress_var.set, "All silos already processed")
                    return
                
                total_silos = len(silos_with_data)
                self.root.after(0, self.simple_response_text.insert, tk.END, f"📊 Found {total_silos} new silos to process (after filtering)\n\n")
                
                processed_count = 0
                successful_count = 0
                failed_count = 0
                
                # Process each silo
                for i, silo in enumerate(silos_with_data, 1):
                    self.root.after(0, self.simple_progress_var.set, f"Processing silo {i}/{total_silos}: {silo['silo_name']}")
                    self.root.after(0, self.simple_response_text.insert, tk.END, 
                        f"[{i}/{total_silos}] Processing: {silo['granary_name']} - {silo['silo_name']}\n")
                    
                    # Fill the form with current silo data
                    self.root.after(0, self.simple_granary_var.set, silo['granary_name'])
                    self.root.after(0, self.simple_silo_var.set, silo['silo_id'])
                    self.root.after(0, self.simple_start_date_var.set, silo['start_date'])
                    self.root.after(0, self.simple_end_date_var.set, silo['end_date'])
                    
                    # Run the data retrieval
                    success = self._run_single_retrieval_sync(silo)
                    
                    if success:
                        successful_count += 1
                        self.root.after(0, self.simple_response_text.insert, tk.END, f"   ✅ Successfully retrieved data for {silo['silo_name']}\n")
                    else:
                        failed_count += 1
                        self.root.after(0, self.simple_response_text.insert, tk.END, f"   ❌ Failed to retrieve data for {silo['silo_name']}\n")
                    
                    processed_count += 1
                    
                    # Small delay between processes
                    import time
                    time.sleep(1)
                
                # Final summary
                self.root.after(0, self.simple_response_text.insert, tk.END, f"\n🎯 Auto Processing Complete!\n")
                self.root.after(0, self.simple_response_text.insert, tk.END, f"   Total processed: {processed_count}\n")
                self.root.after(0, self.simple_response_text.insert, tk.END, f"   Successful: {successful_count}\n")
                self.root.after(0, self.simple_response_text.insert, tk.END, f"   Failed: {failed_count}\n")
                
                self.root.after(0, self.simple_progress_var.set, f"Completed: {successful_count}/{processed_count} successful")
                
            except Exception as e:
                self.root.after(0, self.simple_response_text.insert, tk.END, f"\n❌ Error in auto processing: {e}\n")
                self.root.after(0, self.simple_progress_var.set, "Error in auto processing")
        
        threading.Thread(target=process_all, daemon=True).start()
    
    def _run_single_retrieval_sync(self, silo):
        """Run a single data retrieval synchronously and return success status"""
        try:
            import sys
            import os
            
            # Path to the simple data retrieval script
            script_path = Path(__file__).parent.parent / "simple_data_retrieval.py"
            config_path = Path(__file__).parent.parent.parent / "config" / "production_config.json"
            
            # Build command
            cmd = [
                self.get_python_executable(),
                str(script_path),
                "--granary-name", silo['granary_name'],
                "--silo-id", silo['silo_id'],
                "--start-date", silo['start_date'],
                "--end-date", silo['end_date'],
                "--config", str(config_path),
                "--output-dir", self.simple_output_var.get()
            ]
            
            # Set environment to ensure UTF-8 output
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUNBUFFERED'] = '1'
            
            # Run the command synchronously
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
                env=env,
                timeout=300  # 5 minute timeout per silo
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            self.root.after(0, self.simple_response_text.insert, tk.END, f"   ⏰ Timeout processing {silo['silo_name']}\n")
            return False
        except Exception as e:
            self.root.after(0, self.simple_response_text.insert, tk.END, f"   ❌ Exception processing {silo['silo_name']}: {e}\n")
            return False
    
    def run_simple_retrieval(self):
        """Run simple data retrieval"""
        def run_retrieval():
            try:
                import sys
                import os
                
                self.simple_progress_var.set("Running simple data retrieval...")
                self.simple_response_text.insert(tk.END, f"\n📊 Starting simple data retrieval...\n")
                self.simple_response_text.insert(tk.END, f"Granary: {self.simple_granary_var.get()}\n")
                self.simple_response_text.insert(tk.END, f"Silo ID: {self.simple_silo_var.get()}\n")
                self.simple_response_text.insert(tk.END, f"Date Range: {self.simple_start_date_var.get()} to {self.simple_end_date_var.get()}\n")
                
                # Path to the simple data retrieval script
                script_path = Path(__file__).parent.parent / "simple_data_retrieval.py"
                config_path = Path(__file__).parent.parent.parent / "config" / "production_config.json"
                
                # Build command
                cmd = [
                    self.get_python_executable(),
                    str(script_path),
                    "--granary-name", self.simple_granary_var.get(),
                    "--silo-id", self.simple_silo_var.get(),
                    "--start-date", self.simple_start_date_var.get(),
                    "--end-date", self.simple_end_date_var.get(),
                    "--config", str(config_path),
                    "--output-dir", self.simple_output_var.get()
                ]
                
                self.root.after(0, self.simple_response_text.insert, tk.END, f"Command: {' '.join(cmd)}\n\n")
                
                # Set environment to ensure UTF-8 output
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONUNBUFFERED'] = '1'
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=0,  # Unbuffered
                    universal_newlines=True,
                    cwd=Path(__file__).parent.parent.parent,
                    env=env
                )
                
                # Read output in real-time
                if process.stdout:
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            self.root.after(0, self.simple_response_text.insert, tk.END, output)
                            self.root.after(0, self.simple_response_text.see, tk.END)
                            self.root.after(0, self.root.update_idletasks)  # Force GUI update
                
                return_code = process.poll()
                
                if return_code == 0:
                    self.root.after(0, self.simple_progress_var.set, "Simple retrieval completed successfully")
                    self.root.after(0, self.simple_response_text.insert, tk.END, "\n✅ Simple data retrieval completed successfully!\n")
                else:
                    self.root.after(0, self.simple_progress_var.set, f"Simple retrieval failed (exit code {return_code})")
                    self.root.after(0, self.simple_response_text.insert, tk.END, f"\n❌ Simple retrieval failed with exit code {return_code}\n")
                    
            except Exception as e:
                self.root.after(0, self.simple_response_text.insert, tk.END, f"\n❌ Error: {e}\n")
                self.root.after(0, self.simple_progress_var.set, "Error in simple retrieval")
        
        threading.Thread(target=run_retrieval, daemon=True).start()
    
    def open_simple_output_folder(self):
        """Open the simple retrieval output folder"""
        output_dir = Path(self.simple_output_var.get())
        if output_dir.exists():
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(output_dir)])
            else:
                subprocess.run(["xdg-open", str(output_dir)])
        else:
            messagebox.showwarning("Directory Not Found", f"Output directory does not exist: {output_dir}")
    
    def clear_simple_log(self):
        """Clear the simple retrieval log"""
        self.simple_response_text.delete(1.0, tk.END)
        self.simple_progress_var.set("Ready")

    # ------------------------------------------------------------------
    # BATCH PROCESSING METHODS
    # ------------------------------------------------------------------
    
    def browse_batch_input_folder(self):
        """Browse for batch processing input folder"""
        folder = filedialog.askdirectory(
            title="Select Input Folder for Batch Processing",
            initialdir=self.batch_input_folder_var.get() if self.batch_input_folder_var.get() else "data"
        )
        if folder:
            self.batch_input_folder_var.set(folder)
            self.batch_log_text.insert(tk.END, f"📁 Input folder set to: {folder}\n")
    
    def browse_batch_output_folder(self):
        """Browse for batch processing output folder"""
        folder = filedialog.askdirectory(
            title="Select Output Folder for Batch Processing",
            initialdir=self.batch_output_folder_var.get() if self.batch_output_folder_var.get() else "data"
        )
        if folder:
            self.batch_output_folder_var.set(folder)
            self.batch_log_text.insert(tk.END, f"📂 Output folder set to: {folder}\n")
    
    def browse_batch_config(self):
        """Browse for batch processing config file"""
        config_file = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="config" if Path("config").exists() else "."
        )
        if config_file:
            self.batch_config_var.set(config_file)
            self.batch_log_text.insert(tk.END, f"⚙️ Config file set to: {config_file}\n")
    
    def scan_batch_folder(self):
        """Scan the input folder for files to process"""
        def scan_files():
            try:
                input_folder = Path(self.batch_input_folder_var.get())
                if not input_folder.exists():
                    self.root.after(0, messagebox.showerror, "Error", f"Input folder does not exist: {input_folder}")
                    return
                
                self.root.after(0, self.batch_status_var.set, "Scanning folder...")
                self.root.after(0, self.batch_log_text.insert, tk.END, f"\n🔍 Scanning folder: {input_folder}\n")
                
                # Show selected action
                selected_action = self.batch_action_var.get()
                self.root.after(0, self.batch_log_text.insert, tk.END, f"🎯 Selected action: {selected_action.title()}\n")
                
                # Get file extensions and pattern
                extensions = [ext.strip() for ext in self.batch_file_extensions_var.get().split(",")]
                pattern = self.batch_file_pattern_var.get().strip()
                
                # Find matching files
                all_files = []
                for ext in extensions:
                    if ext.startswith("*"):
                        ext = ext[1:]  # Remove leading *
                    files = list(input_folder.rglob(f"*{ext}"))
                    all_files.extend(files)
                
                # Apply pattern filter if specified
                if pattern:
                    import fnmatch
                    filtered_files = [f for f in all_files if fnmatch.fnmatch(f.name, pattern)]
                else:
                    filtered_files = all_files
                
                # Remove duplicates and sort
                self.batch_files_list = sorted(list(set(filtered_files)))
                
                self.root.after(0, self.batch_log_text.insert, tk.END, f"📊 Found {len(self.batch_files_list)} files to process:\n")
                
                total_size_mb = 0
                large_files = []
                
                for i, file_path in enumerate(self.batch_files_list[:10], 1):  # Show first 10
                    file_size = file_path.stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    total_size_mb += size_mb
                    
                    # Flag large files (>500MB)
                    if size_mb > 500:
                        large_files.append((file_path.name, size_mb))
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   {i}. {file_path.name} ({size_mb:.2f} MB) ⚠️ LARGE FILE\n")
                    else:
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   {i}. {file_path.name} ({size_mb:.2f} MB)\n")
                
                # Calculate total size for remaining files
                for file_path in self.batch_files_list[10:]:
                    file_size = file_path.stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    total_size_mb += size_mb
                    if size_mb > 500:
                        large_files.append((file_path.name, size_mb))
                
                if len(self.batch_files_list) > 10:
                    remaining = len(self.batch_files_list) - 10
                    self.root.after(0, self.batch_log_text.insert, tk.END, f"   ... and {remaining} more files\n")
                
                # Show resource warnings
                self.root.after(0, self.batch_log_text.insert, tk.END, f"\n📊 Total dataset size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)\n")
                
                if large_files:
                    self.root.after(0, self.batch_log_text.insert, tk.END, f"⚠️ Found {len(large_files)} large files (>500MB):\n")
                    for filename, size_mb in large_files[:5]:  # Show first 5 large files
                        self.root.after(0, self.batch_log_text.insert, tk.END, f"   • {filename} ({size_mb:.1f} MB)\n")
                    if len(large_files) > 5:
                        self.root.after(0, self.batch_log_text.insert, tk.END, f"   ... and {len(large_files)-5} more large files\n")
                    
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        "💡 Tip: Large files may cause memory issues. Consider processing them individually.\n")
                
                # Memory usage estimate
                try:
                    import psutil
                    available_memory_gb = psutil.virtual_memory().available / (1024**3)
                    estimated_memory_needed_gb = total_size_mb / 1024 * 3  # Rough estimate: 3x file size
                    
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"💾 Available memory: {available_memory_gb:.1f} GB, Estimated needed: {estimated_memory_needed_gb:.1f} GB\n")
                    
                    if estimated_memory_needed_gb > available_memory_gb * 0.8:
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            "⚠️ WARNING: May not have enough memory for batch processing all files!\n")
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            "💡 Consider processing fewer files at once or adding more RAM.\n")
                except ImportError:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        "💡 Install psutil for memory usage estimates: pip install psutil\n")
                
                if not self.batch_files_list:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "⚠️ No files found matching the criteria\n")
                
                self.root.after(0, self.batch_progress_var.set, f"0/{len(self.batch_files_list)} files")
                self.root.after(0, self.batch_status_var.set, f"Found {len(self.batch_files_list)} files")
                
            except Exception as e:
                self.root.after(0, self.batch_log_text.insert, tk.END, f"❌ Error scanning folder: {e}\n")
                self.root.after(0, self.batch_status_var.set, "Scan failed")
        
        threading.Thread(target=scan_files, daemon=True).start()
    
    def start_batch_processing(self):
        """Start sequential batch processing optimized for massive files with limited resources"""
        if not self.batch_files_list:
            messagebox.showerror("Error", "No files to process. Please scan folder first.")
            return
        
        if self.batch_processing_active:
            messagebox.showwarning("Warning", "Batch processing is already running!")
            return
        
        # Validate that an action is selected
        selected_action = self.batch_action_var.get()
        if not selected_action:
            messagebox.showerror("Error", "Please select a processing action first.")
            return
        
        def run_batch():
            try:
                # Initialize streaming processor for rock-solid sequential processing
                from granarypredict.streaming_processor import MassiveDatasetProcessor, estimate_memory_requirements
                
                self.batch_processing_active = True
                self.batch_current_file_index = 0
                total_files = len(self.batch_files_list)
                
                self.root.after(0, self.batch_status_var.set, f"Running sequential {selected_action}...")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"\n🚀 Starting SEQUENTIAL {selected_action} of {total_files} files\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, "=" * 60 + "\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"🔧 Mode: One file at a time for maximum stability and efficiency\n")
                
                # Initialize streaming processor with conservative settings
                try:
                    max_memory_gb = float(self.batch_max_memory_var.get())
                except ValueError:
                    max_memory_gb = 2.0
                
                # Conservative chunk size for tens of millions of rows - start small and let it adapt
                chunk_size = 50_000  # Start with 50K rows per chunk
                
                processor = MassiveDatasetProcessor(
                    chunk_size=chunk_size,
                    memory_threshold_percent=70.0,  # Conservative threshold
                    backend="auto",  # Will choose best available backend
                    enable_dask=False,  # No parallel processing
                    n_workers=1  # Single worker only
                )
                
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"🔧 Streaming processor: {processor.backend} backend, initial chunk: {chunk_size:,} rows\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"💾 Conservative memory threshold: 70%, Sequential processing only\n\n")
                
                successful_files = 0
                failed_files = 0
                skipped_files = 0
                
                # Process each file sequentially with maximum care
                for i, file_path in enumerate(self.batch_files_list):
                    if not self.batch_processing_active:  # Check if stopped
                        break
                    
                    self.batch_current_file_index = i + 1
                    self.root.after(0, self.batch_progress_var.set, f"{i + 1}/{total_files} files")
                    
                    # Pre-flight checks for each file
                    try:
                        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        file_size_gb = file_size_mb / 1024
                        
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"\n[{i + 1}/{total_files}] Processing: {file_path.name}\n")
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   📊 File size: {file_size_mb:.1f} MB ({file_size_gb:.2f} GB)\n")
                        
                        # Memory check before processing
                        import psutil
                        available_memory_gb = psutil.virtual_memory().available / (1024**3)
                        memory_percent = psutil.virtual_memory().percent
                        
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   💾 Available memory: {available_memory_gb:.1f}GB ({100-memory_percent:.1f}% free)\n")
                        
                        # Skip if file is too large for available memory
                        if file_size_gb > available_memory_gb * 0.5:
                            self.root.after(0, self.batch_log_text.insert, tk.END, 
                                f"   ⚠️ Skipping: File too large for available memory\n")
                            skipped_files += 1
                            continue
                        
                        # Memory cleanup before each file
                        import gc
                        gc.collect()
                        
                        # Process the file using streaming processor
                        success = self._process_single_file_streaming(
                            file_path, 
                            selected_action,
                            processor
                        )
                        
                        if success:
                            successful_files += 1
                            self.root.after(0, self.batch_log_text.insert, tk.END, 
                                f"   ✅ Successfully processed {file_path.name}\n")
                        else:
                            failed_files += 1
                            self.root.after(0, self.batch_log_text.insert, tk.END, 
                                f"   ❌ Failed to process {file_path.name}\n")
                        
                        # Memory cleanup after each file
                        gc.collect()
                        
                        # Brief pause between files for system stability
                        import time
                        time.sleep(1)
                        
                    except Exception as e:
                        failed_files += 1
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   ❌ Error processing {file_path.name}: {str(e)}\n")
                
                # Final summary
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"\n🎯 Sequential {selected_action} completed!\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"   Total files: {total_files}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"   Successful: {successful_files}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"   Failed: {failed_files}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"   Skipped: {skipped_files}\n")
                
                total_processed_rows = getattr(processor, 'processed_rows', 0)
                if total_processed_rows > 0:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"   Total rows processed: {total_processed_rows:,}\n")
                
                self.root.after(0, self.batch_status_var.set, 
                    f"Completed: {successful_files}/{total_files} successful")
                
                self.batch_processing_active = False
                
            except Exception as e:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"\n❌ Critical batch processing error: {e}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"   Falling back to legacy processing...\n")
                # Emergency fallback to legacy processing
                self._run_legacy_batch_processing(selected_action)
                
                self.batch_processing_active = True
                self.batch_current_file_index = 0
                total_files = len(self.batch_files_list)
                
                self.root.after(0, self.batch_status_var.set, f"Initializing massive batch {selected_action}...")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"\n🚀 Starting MASSIVE SCALE batch {selected_action} of {total_files} files\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, "=" * 80 + "\n")
                
                # Create batch processing strategy based on file analysis
                batch_strategy = self._analyze_batch_requirements()
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"� Batch Strategy: {batch_strategy['strategy']}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"🔧 Parallel Workers: {batch_strategy['workers']}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"💾 Memory per Worker: {batch_strategy['memory_per_worker']:.1f}GB\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"📦 Batch Size: {batch_strategy['batch_size']} files\n\n")
                
                if selected_action == "processing":
                    success = self._run_massive_batch_processing(batch_strategy)
                else:
                    # For other actions, use optimized sequential processing
                    success = self._run_optimized_sequential_batch(selected_action, batch_strategy)
                
                if success:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"\n🎯 MASSIVE BATCH {selected_action.upper()} COMPLETED SUCCESSFULLY!\n")
                else:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"\n⚠️ Batch {selected_action} completed with some failures\n")
                
                self.batch_processing_active = False
                
            except Exception as e:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"\n❌ Critical batch processing error: {e}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"   Attempting recovery with simplified processing...\n")
                # Emergency fallback
                self._emergency_fallback_processing(selected_action)
                
        threading.Thread(target=run_batch, daemon=True).start()

    def _analyze_batch_requirements(self):
        """Analyze batch requirements and determine optimal processing strategy"""
        try:
            import psutil
            
            # System resource analysis
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = psutil.cpu_count() or 4  # Fallback to 4 if None
            
            # File analysis
            total_files = len(self.batch_files_list)
            file_sizes = []
            total_size_gb = 0
            
            for file_path in self.batch_files_list[:min(20, total_files)]:  # Sample first 20 files
                try:
                    size_gb = os.path.getsize(file_path) / (1024**3)
                    file_sizes.append(size_gb)
                    total_size_gb += size_gb
                except:
                    file_sizes.append(0.1)  # Assume 100MB if can't read
            
            # Estimate total size for all files
            if file_sizes:
                avg_file_size_gb = sum(file_sizes) / len(file_sizes)
                estimated_total_size_gb = avg_file_size_gb * total_files
            else:
                avg_file_size_gb = 0.5
                estimated_total_size_gb = avg_file_size_gb * total_files
            
            # Determine strategy based on analysis
            try:
                max_memory_per_file = float(self.batch_max_memory_var.get())
            except:
                max_memory_per_file = 2.0
            
            # Calculate optimal workers and batch size
            if estimated_total_size_gb < available_memory_gb * 0.3:
                # Small dataset - can do parallel processing
                strategy = "parallel_in_memory"
                workers = min(cpu_count, max(2, int(available_memory_gb / max_memory_per_file)))
                batch_size = min(50, max(5, total_files // workers))
            elif avg_file_size_gb > 5.0:
                # Very large files - sequential streaming
                strategy = "sequential_streaming"
                workers = 1
                batch_size = 1
            elif total_files > 100:
                # Many files - controlled parallel with limited workers
                strategy = "controlled_parallel"
                workers = min(3, max(2, int(available_memory_gb / (max_memory_per_file * 2))))
                batch_size = min(20, max(3, total_files // (workers * 4)))
            else:
                # Moderate load - standard parallel
                strategy = "standard_parallel"
                workers = min(4, max(2, int(available_memory_gb / max_memory_per_file)))
                batch_size = min(10, max(2, total_files // workers))
            
            return {
                'strategy': strategy,
                'workers': workers,
                'batch_size': batch_size,
                'memory_per_worker': max_memory_per_file,
                'estimated_total_size_gb': estimated_total_size_gb,
                'avg_file_size_gb': avg_file_size_gb,
                'available_memory_gb': available_memory_gb,
                'total_files': total_files
            }
            
        except Exception as e:
            # Fallback to conservative settings
            return {
                'strategy': 'conservative_sequential',
                'workers': 1,
                'batch_size': 1,
                'memory_per_worker': 1.0,
                'estimated_total_size_gb': 10.0,
                'avg_file_size_gb': 1.0,
                'available_memory_gb': 4.0,
                'total_files': len(self.batch_files_list),
                'error': str(e)
            }

    def _run_massive_batch_processing(self, batch_strategy):
        """Run massive scale batch processing with optimized resource management"""
        try:
            from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
            from granarypredict.streaming_processor import MassiveDatasetProcessor
            import multiprocessing as mp
            
            total_files = len(self.batch_files_list)
            workers = batch_strategy['workers']
            batch_size = batch_strategy['batch_size']
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"🔧 Initializing {workers} processing workers for massive batch processing...\n")
            
            # Create progress tracking
            completed_files = 0
            successful_files = 0
            failed_files = 0
            
            # Process files in batches to avoid memory exhaustion
            file_batches = [self.batch_files_list[i:i + batch_size] 
                           for i in range(0, total_files, batch_size)]
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"📦 Split {total_files} files into {len(file_batches)} batches of ~{batch_size} files each\n\n")
            
            # Use ThreadPoolExecutor for I/O bound operations (better for file processing)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for batch_idx, file_batch in enumerate(file_batches):
                    if not self.batch_processing_active:
                        break
                    
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"🔄 Processing batch {batch_idx + 1}/{len(file_batches)} ({len(file_batch)} files)...\n")
                    
                    # Submit batch for parallel processing
                    future_to_file = {
                        executor.submit(self._process_single_file_optimized, file_path, batch_strategy): file_path
                        for file_path in file_batch
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_file):
                        if not self.batch_processing_active:
                            break
                            
                        file_path = future_to_file[future]
                        completed_files += 1
                        
                        try:
                            success = future.result(timeout=1800)  # 30 minute timeout per file
                            if success:
                                successful_files += 1
                                self.root.after(0, self.batch_log_text.insert, tk.END, 
                                    f"   ✅ [{completed_files}/{total_files}] {file_path.name}\n")
                            else:
                                failed_files += 1
                                self.root.after(0, self.batch_log_text.insert, tk.END, 
                                    f"   ❌ [{completed_files}/{total_files}] {file_path.name} (processing failed)\n")
                        except Exception as e:
                            failed_files += 1
                            self.root.after(0, self.batch_log_text.insert, tk.END, 
                                f"   💥 [{completed_files}/{total_files}] {file_path.name} (error: {str(e)[:50]}...)\n")
                        
                        # Update progress
                        self.root.after(0, self.batch_progress_var.set, f"{completed_files}/{total_files} files")
                    
                    # Memory cleanup between batches
                    if batch_idx < len(file_batches) - 1:  # Not the last batch
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   🧹 Cleaning up memory before next batch...\n")
                        import gc
                        gc.collect()
                        import time
                        time.sleep(2)  # Let memory settle
            
            # Final summary
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"\n📊 MASSIVE BATCH PROCESSING SUMMARY:\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"   Strategy: {batch_strategy['strategy']}\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"   Workers: {workers}\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"   Batches: {len(file_batches)}\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"   Total Files: {total_files}\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"   Successful: {successful_files}\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"   Failed: {failed_files}\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"   Success Rate: {(successful_files/total_files*100):.1f}%\n")
            
            self.root.after(0, self.batch_status_var.set, 
                f"Massive batch completed: {successful_files}/{total_files} successful")
            
            return successful_files > 0
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"❌ Critical error in massive batch processing: {e}\n")
            return False

    def _process_single_file_optimized(self, file_path, batch_strategy):
        """Optimized single file processing for massive batch operations"""
        try:
            from granarypredict.streaming_processor import MassiveDatasetProcessor
            import tempfile
            import shutil
            
            # Create processor with batch-optimized settings
            processor = MassiveDatasetProcessor(
                chunk_size=max(100_000, int(batch_strategy['memory_per_worker'] * 1024**3 / (8 * 50))),
                memory_threshold_percent=85.0,  # Higher threshold for batch processing
                backend="auto",
                enable_dask=False,  # Disable Dask in batch mode to avoid conflicts
                n_workers=1  # Single worker per file in batch mode
            )
            
            output_folder = Path(self.batch_output_folder_var.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Determine granary name
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            
            # Create final output path
            final_output = output_folder / f"{granary_name}_processed.parquet"
            
            # Skip if file already exists and is newer than source
            if final_output.exists():
                try:
                    output_time = os.path.getmtime(final_output)
                    source_time = os.path.getmtime(file_path)
                    if output_time > source_time:
                        return True  # Already processed
                except:
                    pass  # Continue processing if we can't check times
            
            # Use streaming processor for efficient processing
            success = processor.process_massive_features(
                file_path=file_path,
                output_path=final_output
                # Uses default optimized feature functions
            )
            
            if success and final_output.exists():
                # Verify output file is valid
                try:
                    import pandas as pd
                    df_test = pd.read_parquet(final_output, nrows=1)
                    return len(df_test.columns) > 0
                except:
                    return False
            
            return success
            
        except Exception as e:
            # Log error but don't crash the batch
            return False

    def _run_optimized_sequential_batch(self, selected_action, batch_strategy):
        """Run optimized sequential batch processing for non-processing actions"""
        try:
            total_files = len(self.batch_files_list)
            successful_files = 0
            failed_files = 0
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"🔄 Running optimized sequential {selected_action} processing...\n")
            
            # Pre-allocate resources
            from granarypredict.streaming_processor import MassiveDatasetProcessor
            processor = MassiveDatasetProcessor(
                chunk_size=min(500_000, int(batch_strategy['memory_per_worker'] * 1024**3 / (8 * 100))),
                memory_threshold_percent=75.0,
                backend="auto",
                enable_dask=True,
                n_workers=batch_strategy['workers']
            )
            
            # Process files with optimized resource management
            for i, file_path in enumerate(self.batch_files_list):
                if not self.batch_processing_active:
                    break
                
                self.batch_current_file_index = i + 1
                self.root.after(0, self.batch_progress_var.set, f"{i + 1}/{total_files} files")
                
                try:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"[{i + 1}/{total_files}] Processing {file_path.name}...\n")
                    
                    # Use optimized streaming methods
                    if selected_action == "sorting":
                        success = self._stream_sorting_optimized(file_path, processor)
                    elif selected_action == "training":
                        success = self._stream_training_optimized(file_path, processor)
                    elif selected_action == "forecasting":
                        success = self._stream_forecasting_optimized(file_path, processor)
                    else:
                        success = False
                    
                    if success:
                        successful_files += 1
                        self.root.after(0, self.batch_log_text.insert, tk.END, f"   ✅ Success\n")
                    else:
                        failed_files += 1
                        self.root.after(0, self.batch_log_text.insert, tk.END, f"   ❌ Failed\n")
                        
                except Exception as e:
                    failed_files += 1
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"   💥 Error: {str(e)[:100]}...\n")
                
                # Periodic cleanup every 10 files
                if i % 10 == 9:
                    import gc
                    gc.collect()
            
            self.root.after(0, self.batch_status_var.set, 
                f"Completed {selected_action}: {successful_files}/{total_files} successful")
            
            return successful_files > 0
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"❌ Error in optimized sequential processing: {e}\n")
            return False

    def _emergency_fallback_processing(self, selected_action):
        """Emergency fallback when all optimized methods fail"""
        try:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"🚨 EMERGENCY FALLBACK: Processing files one-by-one with maximum safety...\n")
            
            successful_files = 0
            total_files = len(self.batch_files_list)
            
            for i, file_path in enumerate(self.batch_files_list):
                if not self.batch_processing_active:
                    break
                
                try:
                    # Most basic processing possible
                    if selected_action == "processing":
                        success = self._basic_emergency_processing(file_path)
                    else:
                        success = self._process_single_file_legacy(file_path, selected_action)
                    
                    if success:
                        successful_files += 1
                        
                    self.root.after(0, self.batch_progress_var.set, f"{i + 1}/{total_files} files")
                    
                    # Aggressive memory cleanup
                    import gc
                    gc.collect()
                    import time
                    time.sleep(1)
                    
                except Exception:
                    continue  # Skip problematic files
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"🚨 Emergency processing completed: {successful_files}/{total_files} successful\n")
            
            self.batch_processing_active = False
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"💥 Emergency fallback failed: {e}\n")
            self.batch_processing_active = False

    def _basic_emergency_processing(self, file_path):
        """Most basic processing possible for emergency situations"""
        try:
            import pandas as pd
            
            output_folder = Path(self.batch_output_folder_var.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            output_file = output_folder / f"{granary_name}_processed.parquet"
            
            # Read file with minimal processing
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=100000)  # Limit rows for safety
            else:
                df = pd.read_parquet(file_path)
                if len(df) > 100000:
                    df = df.head(100000)  # Limit rows for safety
            
            # Minimal processing: just clean and save
            df_clean = df.dropna()
            if len(df_clean) > 0:
                df_clean.to_parquet(output_file, index=False)
                return True
            
            return False
            
        except Exception:
            return False

    def _stream_sorting_optimized(self, file_path, processor):
        """Optimized streaming sorting for batch operations"""
        try:
            output_folder = Path(self.batch_output_folder_var.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Use streaming processor for sorting
            temp_output = output_folder / f"{file_path.stem}_sorted.parquet"
            
            def sorting_features(chunk_df):
                if 'granary_name' in chunk_df.columns:
                    return chunk_df.sort_values(['granary_name', 'detection_time'] 
                                              if 'detection_time' in chunk_df.columns else ['granary_name'])
                return chunk_df
            
            success = processor.process_massive_features(
                file_path=file_path,
                output_path=temp_output,
                feature_functions=[sorting_features]
            )
            
            return success and temp_output.exists()
            
        except Exception:
            return False

    def _stream_training_optimized(self, file_path, processor):
        """Optimized streaming training for batch operations"""
        try:
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            
            # Ensure processed data exists
            processed_file = self._ensure_processed_data(file_path, processor)
            if not processed_file:
                return False
            
            # Use existing training logic but optimized
            return self._stream_training(processed_file, granary_name)
            
        except Exception:
            return False

    def _stream_forecasting_optimized(self, file_path, processor):
        """Optimized streaming forecasting for batch operations"""
        try:
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            
            # Ensure processed data exists
            processed_file = self._ensure_processed_data(file_path, processor)
            if not processed_file:
                return False
            
            # Use existing forecasting logic but optimized
            return self._stream_forecasting(processed_file, granary_name)
            
        except Exception:
            return False
    
    def _process_single_file_streaming(self, file_path, action, processor):
        """Process a single file using the streaming processor for efficient large file handling"""
        try:
            output_folder = Path(self.batch_output_folder_var.get())
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create temporary output path for streaming processing
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="siloflow_batch_"))
            temp_output = temp_dir / f"{file_path.stem}_streaming_output.parquet"
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      🔄 Using streaming processor for {action}...\n")
            
            success = False
            
            if action == "sorting":
                # For sorting, use streaming processor to efficiently organize data
                success = self._stream_sorting(file_path, output_folder, processor)
                
            elif action == "processing":
                # Use streaming processor for data cleaning and feature engineering
                success = self._stream_processing(file_path, output_folder, processor, temp_output)
                
            elif action == "training":
                # For training, first ensure data is processed, then train
                processed_file = self._ensure_processed_data(file_path, processor)
                if processed_file:
                    success = self._stream_training(processed_file, file_path.stem)
                    
            elif action == "forecasting":
                # For forecasting, ensure model exists and data is processed
                processed_file = self._ensure_processed_data(file_path, processor)
                if processed_file:
                    success = self._stream_forecasting(processed_file, file_path.stem)
            
            # Cleanup temporary directory
            import shutil
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"      ⚠️ Could not cleanup temp dir: {e}\n")
            
            return success
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      ❌ Streaming processing error: {e}\n")
            # Fallback to legacy processing
            return self._process_single_file_legacy(file_path, action)
    
    def _stream_sorting(self, file_path, output_folder, processor):
        """Use streaming processor for efficient data sorting"""
        try:
            # Define sorting feature function for streaming
            def sorting_features(chunk_df):
                # Basic granary identification and organization
                if 'granary_name' in chunk_df.columns:
                    # Group by granary for efficient sorting
                    return chunk_df.sort_values(['granary_name', 'detection_time'] if 'detection_time' in chunk_df.columns else ['granary_name'])
                return chunk_df
            
            # Process with streaming
            temp_output = output_folder / f"{file_path.stem}_sorted.parquet"
            success = processor.process_massive_features(
                file_path=file_path,
                output_path=temp_output,
                feature_functions=[sorting_features]
            )
            
            if success:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ✅ Streaming sort completed: {temp_output}\n")
            
            return success
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      ❌ Streaming sort failed: {e}\n")
            return False
    
    def _stream_processing(self, file_path, output_folder, processor, temp_output):
        """Use streaming processor for data processing with feature engineering"""
        try:
            # Create comprehensive processing output
            final_output = output_folder / f"{file_path.stem}_processed.parquet"
            
            # Use the streaming processor's built-in feature engineering
            success = processor.process_massive_features(
                file_path=file_path,
                output_path=final_output
                # Uses default feature functions: time features, lags, rolling features
            )
            
            if success:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ✅ Streaming processing completed: {final_output}\n")
                
                # Show processing stats
                if final_output.exists():
                    file_size = final_output.stat().st_size / (1024**2)
                    rows_processed = getattr(processor, 'processed_rows', 0)
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"      📊 Output: {file_size:.1f}MB, {rows_processed:,} rows processed\n")
            
            return success
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      ❌ Streaming processing failed: {e}\n")
            return False
    
    def _ensure_processed_data(self, file_path, processor):
        """Ensure data is processed for training/forecasting, using streaming if needed"""
        try:
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            processed_file = processed_dir / f"{granary_name}_processed.parquet"
            
            # Check if processed file already exists and is recent
            if processed_file.exists():
                file_time = os.path.getmtime(file_path)
                processed_time = os.path.getmtime(processed_file)
                if processed_time > file_time:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"      ℹ️ Using existing processed file: {processed_file}\n")
                    return processed_file
            
            # Need to process the data using streaming
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      🔄 Processing data with streaming processor...\n")
            
            success = processor.process_massive_features(
                file_path=file_path,
                output_path=processed_file
            )
            
            if success and processed_file.exists():
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ✅ Data processed for ML operations: {processed_file}\n")
                return processed_file
            else:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ❌ Failed to process data for ML operations\n")
                return None
                
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      ❌ Error ensuring processed data: {e}\n")
            return None
    
    def _stream_training(self, processed_file, granary_name):
        """Train model using processed data"""
        try:
            # Use the existing CLI training approach but with better logging
            import subprocess
            script_path = Path(__file__).parent.parent.parent / "granary_pipeline.py"
            
            # Get tuning options
            use_tuning = self.batch_tune_var.get()
            use_gpu = self.batch_gpu_var.get()
            trials = self.batch_trials_var.get()
            timeout = self.batch_timeout_var.get()
            
            cmd = [
                self.get_python_executable(), str(script_path),
                "train", "--granary", granary_name
            ]
            
            if use_tuning:
                cmd.extend(["--tune", "--trials", str(trials), "--timeout", str(timeout)])
            
            if use_gpu:
                cmd.append("--gpu")
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      🤖 Training model for {granary_name}...\n")
            
            working_dir = Path(__file__).parent.parent.parent
            env = os.environ.copy()
            env["SILOFLOW_TRAIN_ONLY"] = "1"
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=working_dir, env=env)
            
            if result.returncode == 0:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ✅ Model training completed for {granary_name}\n")
                return True
            else:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ❌ Model training failed: {result.stderr}\n")
                return False
                
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      ❌ Training error: {e}\n")
            return False
    
    def _stream_forecasting(self, processed_file, granary_name):
        """Generate forecasts using processed data"""
        try:
            import subprocess
            script_path = Path(__file__).parent.parent.parent / "granary_pipeline.py"
            
            use_gpu = self.batch_gpu_var.get()
            
            cmd = [
                self.get_python_executable(), str(script_path),
                "forecast", "--granary", granary_name, "--horizon", "7"
            ]
            
            if use_gpu:
                cmd.append("--gpu")
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      🔮 Generating forecasts for {granary_name}...\n")
            
            working_dir = Path(__file__).parent.parent.parent
            env = os.environ.copy()
            env["SILOFLOW_FORECAST_ONLY"] = "1"
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=working_dir, env=env)
            
            if result.returncode == 0:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ✅ Forecasting completed for {granary_name}\n")
                return True
            else:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ❌ Forecasting failed: {result.stderr}\n")
                return False
                
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      ❌ Forecasting error: {e}\n")
            return False

    def _process_single_file_legacy(self, file_path, action):
        """Legacy processing method as fallback"""
        try:
            # Convert action to legacy flags
            run_sorting = (action == "sorting")
            run_processing = (action == "processing") 
            run_training = (action == "training")
            run_forecasting = (action == "forecasting")
            
            return self._process_single_file(
                file_path, 
                run_sorting, 
                run_processing, 
                run_training, 
                run_forecasting
            )
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"      ❌ Legacy processing failed: {e}\n")
            return False

    def _run_legacy_batch_processing(self, selected_action):
        """Fallback to the original batch processing method"""
        try:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"🔄 Running legacy batch processing for {selected_action}...\n")
            
            # Set action flags based on selection
            run_sorting = (selected_action == "sorting")
            run_processing = (selected_action == "processing")
            run_training = (selected_action == "training")
            run_forecasting = (selected_action == "forecasting")
            
            successful_files = 0
            failed_files = 0
            
            # Process each file with legacy method
            for i, file_path in enumerate(self.batch_files_list):
                if not self.batch_processing_active:
                    break
                
                try:
                    success = self._process_single_file(
                        file_path, 
                        run_sorting, 
                        run_processing, 
                        run_training, 
                        run_forecasting
                    )
                    
                    if success:
                        successful_files += 1
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   ✅ Legacy processed {file_path.name}\n")
                    else:
                        failed_files += 1
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"   ❌ Legacy failed {file_path.name}\n")
                        
                except Exception as e:
                    failed_files += 1
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"   ❌ Legacy error {file_path.name}: {e}\n")
            
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"🎯 Legacy processing completed: {successful_files}/{len(self.batch_files_list)} successful\n")
            
            self.batch_processing_active = False
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"❌ Legacy processing error: {e}\n")
            self.batch_processing_active = False

    def _process_single_file(self, file_path, run_sorting, run_processing, run_training, run_forecasting):
        """Process a single file through the selected pipeline step"""
        try:
            config_path = Path(self.batch_config_var.get())
            output_folder = Path(self.batch_output_folder_var.get())
            
            # Ensure output folder exists
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Don't create file-specific output folders anymore
            # All CLI commands (sort, process, train, forecast) use standard system paths
            file_output = output_folder
            
            # Log info about where model/forecast results will go if those steps are enabled
            if run_training or run_forecasting:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"      ℹ️ Note: Models will be saved to 'models/' directory and forecasts to 'forecasts/' directory\n")
            
            success = False
            
            # Only one action will be True at a time now
            # Step 1: Sorting (if selected)
            if run_sorting:
                self.root.after(0, self.batch_log_text.insert, tk.END, "      🔤 Running data sorting...\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    "         ℹ️ Sorting will split input file by granary into separate .parquet files in the output folder.\n")
                success = self._run_sorting_step(file_path, output_folder, config_path)
                if success:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ✅ Sorting completed\n")
                else:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ❌ Sorting failed\n")

            # Step 2: Processing (if selected)
            elif run_processing:
                self.root.after(0, self.batch_log_text.insert, tk.END, "      ⚙️ Running data processing...\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, "         ℹ️ Output will be saved as .parquet format for better performance\n")
                success = self._run_processing_step(file_path, output_folder, config_path)
                if success:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ✅ Processing completed\n")
                else:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ❌ Processing failed\n")
            
            # Step 3: Training (if selected)
            elif run_training:
                self.root.after(0, self.batch_log_text.insert, tk.END, "      🤖 Running model training...\n")
                success = self._run_training_step(file_path, file_output, config_path)
                if success:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ✅ Training completed\n")
                else:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ❌ Training failed\n")
            
            # Step 4: Forecasting (if selected)
            elif run_forecasting:
                self.root.after(0, self.batch_log_text.insert, tk.END, "      🔮 Running forecasting...\n")
                success = self._run_forecasting_step(file_path, file_output, config_path)
                if success:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ✅ Forecasting completed\n")
                else:
                    self.root.after(0, self.batch_log_text.insert, tk.END, "         ❌ Forecasting failed\n")
            
            return success
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, f"      ❌ Pipeline error: {e}\n")
            return False
    
    def _run_sorting_step(self, file_path, output_folder, config_path):
        """Run the sorting (ingest) step using the CLI pipeline - pure data ingestion only"""
        try:
            import subprocess
            script_path = Path(__file__).parent.parent.parent / "granary_pipeline.py"
            
            # Set working directory to the service directory so data/granaries path is correct
            working_dir = Path(__file__).parent.parent.parent
            
            # SORTING ONLY: Just ingest the data, no training or ML operations
            cmd = [
                self.get_python_executable(), str(script_path),
                "ingest",
                "--input", str(file_path)
            ]
            
            # Set environment to avoid ML library initialization warnings
            env = os.environ.copy()
            env['SILOFLOW_INGEST_ONLY'] = '1'  # Signal to avoid ML library imports
            env['SILOFLOW_DISABLE_PARALLEL'] = '1'  # Disable parallel processing for batch operations
            env['PYTHONWARNINGS'] = 'ignore::UserWarning'  # Suppress harmless warnings
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=working_dir, env=env)
            
            # Check if the operation was actually successful by looking at the output content
            # The operation might return non-zero exit code due to warnings, but still succeed
            success_indicators = [
                "Saved Parquet file:",
                "Compression ratio:",
                "Ingested and sorted data for granaries:"
            ]
            
            actual_success = any(indicator in result.stdout for indicator in success_indicators)
            
            if result.returncode != 0:
                # Filter out harmless warnings that don't indicate actual failure
                stderr_lines = result.stderr.split('\n') if result.stderr else []
                filtered_errors = []
                
                for line in stderr_lines:
                    # Skip harmless warnings that don't indicate failure
                    if any(warning in line for warning in [
                        "Only training set found, disabling early stopping",
                        "missing ScriptRunContext",
                        "WARNING streamlit.runtime",
                        "lightgbm.callback",
                        "UserWarning"
                    ]):
                        continue
                    if line.strip():  # Only add non-empty lines
                        filtered_errors.append(line)
                
                if filtered_errors and not actual_success:
                    # Only report as error if there are real errors AND no success indicators
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"         ❌ Sorting CLI error: {chr(10).join(filtered_errors)}\n")
                elif filtered_errors:
                    # Log warnings but don't fail the operation
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"         ⚠️ Sorting warnings (operation successful): {chr(10).join(filtered_errors[:2])}\n")
            
            if result.stdout.strip():
                # Log the CLI output to show which granaries were processed
                self.root.after(0, self.batch_log_text.insert, tk.END, f"         📊 {result.stdout.strip()}\n")
            
            # Show where files are actually written
            data_granaries_path = working_dir / "data" / "granaries"
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"         📁 Output files written to: {data_granaries_path}\n")
            
            # Return success if we have success indicators OR return code is 0
            return actual_success or (result.returncode == 0)
            
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Sorting CLI error: {e}\n")
            return False
    
    def _run_processing_step(self, file_path, output_folder, config_path):
        """Run the preprocessing step using the CLI pipeline with resource management"""
        try:
            import subprocess
            import psutil
            import os
            import time
            
            # Check available memory before processing
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            file_size_gb = os.path.getsize(file_path) / (1024**3)
            
            # Get user-configured memory limit
            try:
                max_memory_per_file_gb = float(self.batch_max_memory_var.get())
            except ValueError:
                max_memory_per_file_gb = 2.0  # Default if not a valid number
                
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"         📊 Memory check: {available_memory_gb:.1f}GB available, file size: {file_size_gb:.2f}GB, limit: {max_memory_per_file_gb:.1f}GB\n")
            
            # Skip processing if file is too large relative to available memory or limit
            skip_file = False
            skip_reason = ""
            
            if file_size_gb > max_memory_per_file_gb:
                skip_file = True
                skip_reason = f"exceeds memory limit of {max_memory_per_file_gb:.1f}GB"
            elif file_size_gb > available_memory_gb * 0.3:
                skip_file = True
                skip_reason = f"insufficient available memory ({available_memory_gb:.1f}GB)"
                
            if skip_file:
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"         ⚠️ Skipping large file ({file_size_gb:.2f}GB) - {skip_reason}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END, 
                    f"         💡 Consider processing this file separately or increasing memory limit\n")
                return False
            
            script_path = Path(__file__).parent.parent.parent / "granary_pipeline.py"
            
            # Extract granary name from file name for consistency
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            
            # Create the processed directory in the standard location expected by train/forecast
            processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save in the standard location for train/forecast to find
            processed_file = processed_dir / f"{granary_name}_processed.parquet"
            
            # Also save a copy to the user-selected output folder for reference
            user_output_file = output_folder / f"{file_path.stem}_processed.parquet"
            
            # Create command with resource limits - save to standard location first
            cmd = [
                self.get_python_executable(), str(script_path),
                "preprocess",
                "--input", str(file_path),
                "--output", str(processed_file)
            ]
            
            # Log where files will be saved
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"         ℹ️ Processed file will be saved to: {processed_file}\n")
            
            # Log the command being executed for debugging
            self.root.after(0, self.batch_log_text.insert, tk.END, f"         🔧 Starting preprocessing with command:\n")
            self.root.after(0, self.batch_log_text.insert, tk.END, f"         🔧 {' '.join(cmd)}\n")
            
            # Set environment to avoid ML operations and warnings
            env = os.environ.copy()
            env['SILOFLOW_PREPROCESS_ONLY'] = '1'  # Signal to avoid ML operations
            env['SILOFLOW_DISABLE_PARALLEL'] = '1'  # Disable parallel processing for batch operations
            env['PYTHONWARNINGS'] = 'ignore'  # Suppress all warnings
            env['PYTHONUNBUFFERED'] = '1'  # Ensure output is not buffered
            
            # Set working directory
            working_dir = Path(__file__).parent.parent.parent
            
            # Use Popen for real-time output streaming instead of subprocess.run
            try:
                self.root.after(0, self.batch_log_text.insert, tk.END, f"         🔧 Starting preprocessing with real-time output...\n")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True,
                    env=env,
                    cwd=working_dir
                )
                
                # Read output in real-time
                output_lines = []
                start_time = time.time()
                last_update = start_time
                
                while True:
                    if process.stdout is None:
                        break
                    output = process.stdout.readline()
                    
                    if output == '' and process.poll() is not None:
                        break
                    
                    if output:
                        output_lines.append(output.strip())
                        # Show real-time output to user
                        self.root.after(0, self.batch_log_text.insert, tk.END, f"         📋 {output.strip()}\n")
                        last_update = time.time()
                    
                    # Timeout removed - allow unlimited processing time
                    current_time = time.time()
                    # No timeout check - process can run indefinitely
                    
                    # Show progress indicator every 30 seconds if no output
                    if current_time - last_update > 30:
                        elapsed_minutes = int((current_time - start_time) / 60)
                        self.root.after(0, self.batch_log_text.insert, tk.END, f"         ⏳ Still processing... ({elapsed_minutes}m elapsed)\n")
                        last_update = current_time
                    
                    # Small delay to prevent overwhelming the GUI
                    time.sleep(0.1)
                
                return_code = process.poll()
                self.root.after(0, self.batch_log_text.insert, tk.END, f"         📊 Process completed with return code: {return_code}\n")
                
                # Check if processing was successful
                success = return_code == 0 and processed_file.exists()
                
                if not success and return_code == 0:
                    self.root.after(0, self.batch_log_text.insert, tk.END, f"         ⚠️ Process completed but output file not found: {processed_file}\n")
                
            except Exception as e:
                self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Error running subprocess: {e}\n")
                return False
            
            if return_code != 0:
                self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Preprocessing failed with return code: {return_code}\n")
                return False
            else:
                # Copy the file to the user-selected output location for reference
                import shutil
                if processed_file.exists() and processed_file != user_output_file:
                    try:
                        shutil.copy(processed_file, user_output_file)
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"         📋 Copy saved to user output: {user_output_file}\n")
                    except Exception as copy_error:
                        self.root.after(0, self.batch_log_text.insert, tk.END, 
                            f"         ⚠️ Could not copy to user output: {copy_error}\n")
                
                # Verify the file was actually created
                if processed_file.exists():
                    file_size = processed_file.stat().st_size / (1024**2)  # MB
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"         ✅ Processed file created successfully ({file_size:.1f} MB)\n")
                    return True
                else:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"         ❌ Output file was not created: {processed_file}\n")
                    return False
                
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Preprocess CLI error: {e}\n")
            return False
    
    def _basic_data_processing(self, file_path, output_folder):
        """Basic data processing fallback"""
        try:
            if file_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                
                # Basic processing: remove nulls, normalize column names
                df_processed = df.dropna()
                df_processed.columns = [col.lower().replace(' ', '_') for col in df_processed.columns]
                
                # Save processed file as parquet
                processed_file = output_folder / f"{file_path.stem}_processed.parquet"
                df_processed.to_parquet(processed_file, index=False)
                return True
            
            elif file_path.suffix.lower() == '.parquet':
                import pandas as pd
                df = pd.read_parquet(file_path)
                
                # Basic processing: remove nulls, normalize column names
                df_processed = df.dropna()
                df_processed.columns = [col.lower().replace(' ', '_') for col in df_processed.columns]
                
                # Save processed file
                processed_file = output_folder / f"{file_path.stem}_processed.parquet"
                df_processed.to_parquet(processed_file, index=False)
                return True
            
            return False
        except Exception:
            return False
    
    def _run_training_step(self, file_path, output_folder, config_path):
        """Run the training step using the CLI pipeline - model will be saved directly to models/ directory"""
        try:
            import subprocess
            script_path = Path(__file__).parent.parent.parent / "granary_pipeline.py"
            
            # Extract granary name from file name
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            
            # Log the granary name being used
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"         ℹ️ Training model for granary: {granary_name}\n")
            
            # Create models directory if it doesn't exist
            models_dir = Path(__file__).parent.parent.parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Log where model will be saved
            expected_model_path = models_dir / f"{granary_name}_forecast_model.joblib"
            self.root.after(0, self.batch_log_text.insert, tk.END,
                f"         ℹ️ Model will be saved to: {expected_model_path}\n")
            
            # Check if there's a processed file
            processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
            has_processed_file = False
            if processed_dir.exists():
                parquet_files = list(processed_dir.glob(f"*{granary_name}*.parquet"))
                csv_files = list(processed_dir.glob(f"*{granary_name}*.csv"))
                has_processed_file = len(parquet_files) > 0 or len(csv_files) > 0
                
                if has_processed_file:
                    self.root.after(0, self.batch_log_text.insert, tk.END,
                        f"         ℹ️ Found processed file for granary: {granary_name}\n")
                else:
                    self.root.after(0, self.batch_log_text.insert, tk.END,
                        f"         ⚠️ No processed file found for granary: {granary_name}\n")
            
            # Get tuning options from GUI
            use_tuning = self.batch_tune_var.get()
            use_gpu = self.batch_gpu_var.get()
            trials = self.batch_trials_var.get()
            timeout = self.batch_timeout_var.get()
            
            # Run the CLI command with optional Optuna hyperparameter tuning
            cmd = [
                self.get_python_executable(), str(script_path),
                "train",
                "--granary", granary_name
            ]
            
            if use_tuning:
                cmd.extend([
                    "--tune",  # Enable Optuna hyperparameter tuning
                    "--trials", str(trials),
                    "--timeout", str(timeout)
                ])
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         🔍 Using Optuna hyperparameter tuning ({trials} trials, {timeout}s timeout)\n")
            else:
                cmd.append("--no-tune")
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         ⚡ Using fixed parameters (no tuning)\n")
            
            # Add GPU parameter
            if use_gpu:
                cmd.append("--gpu")
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         � GPU acceleration enabled for training\n")
            else:
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         💻 CPU-only training (GPU disabled by user)\n")
            
            # Set working directory to ensure paths are correct
            working_dir = Path(__file__).parent.parent.parent
            
            # Set environment variable to disable folder creation during training
            env = os.environ.copy()
            env["SILOFLOW_NO_SUBFOLDER_CREATION"] = "1"
            env["SILOFLOW_TRAIN_ONLY"] = "1"  # Signal this is training-only mode
            env["SILOFLOW_DISABLE_PARALLEL"] = "1"  # Disable parallel processing for batch operations
            
            # Calculate timeout based on tuning settings (with buffer)
            process_timeout = 900 if use_tuning else 600  # 15 min for tuning, 10 min for fixed params
            if use_tuning:
                try:
                    process_timeout = max(900, int(timeout) + 300)  # Tuning timeout + 5 min buffer
                except ValueError:
                    process_timeout = 900
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=process_timeout, cwd=working_dir, env=env)
            if result.returncode != 0:
                self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Training CLI error: {result.stderr}\n")
            else:
                # Log the CLI output
                if result.stdout.strip():
                    self.root.after(0, self.batch_log_text.insert, tk.END, f"         📊 {result.stdout.strip()}\n")
                # Show where model file is saved
                model_path = models_dir / f"{granary_name}_forecast_model.joblib"
                if model_path.exists():
                    self.root.after(0, self.batch_log_text.insert, tk.END,
                        f"         📁 Model file saved to: {model_path}\n")
            return result.returncode == 0
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Training CLI error: {e}\n")
            return False
    
    def _run_forecasting_step(self, file_path, output_folder, config_path):
        """Run the forecasting step using the CLI pipeline - forecasts will be saved directly to forecasts/ directory"""
        try:
            import subprocess
            script_path = Path(__file__).parent.parent.parent / "granary_pipeline.py"
            
            # Extract granary name from file name
            granary_name = file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem
            
            # Log the granary name being used
            self.root.after(0, self.batch_log_text.insert, tk.END, 
                f"         ℹ️ Generating forecasts for granary: {granary_name}\n")
            
            # Check if model file exists
            models_dir = Path(__file__).parent.parent.parent / "models"
            model_path = models_dir / f"{granary_name}_forecast_model.joblib"
            
            if not model_path.exists():
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         ⚠️ No model file found for granary: {granary_name}\n")
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         ℹ️ Expected model file: {model_path}\n")
            
            # Create forecasts directory if it doesn't exist
            forecasts_dir = Path(__file__).parent.parent.parent / "forecasts"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            
            # Log where forecasts will be saved
            self.root.after(0, self.batch_log_text.insert, tk.END,
                f"         ℹ️ Forecasts will be saved to: {forecasts_dir}\n")
            
            # Get GPU setting from GUI
            use_gpu = self.batch_gpu_var.get()
            
            # Run the CLI command with proper working directory
            cmd = [
                self.get_python_executable(), str(script_path),
                "forecast",
                "--granary", granary_name,
                "--horizon", "7"
            ]
            
            # Add GPU parameter
            if use_gpu:
                cmd.append("--gpu")
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         🚀 GPU acceleration enabled for forecasting\n")
            else:
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         💻 CPU-only forecasting (GPU disabled by user)\n")
            
            # Set working directory to ensure paths are correct
            working_dir = Path(__file__).parent.parent.parent
            
            # Set environment variables to ensure only forecasting operations
            env = os.environ.copy()
            env["SILOFLOW_FORECAST_ONLY"] = "1"
            env["SILOFLOW_NO_SUBFOLDER_CREATION"] = "1"
            env["SILOFLOW_DISABLE_PARALLEL"] = "1"  # Disable parallel processing for batch operations
            env["PYTHONWARNINGS"] = "ignore"  # Suppress warnings during forecasting
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=working_dir, env=env)
            
            # Filter out harmless warnings in stderr if they exist
            if result.stderr:
                filtered_stderr = []
                for line in result.stderr.split('\n'):
                    if line.strip() and not any(warning in line.lower() for warning in [
                        'lightgbm', 'lgb', 'warning', 'deprecation', 'future'
                    ]):
                        filtered_stderr.append(line)
                if filtered_stderr:
                    self.root.after(0, self.batch_log_text.insert, tk.END, 
                        f"         ⚠️ Forecasting warnings: {chr(10).join(filtered_stderr)}\n")
            
            if result.returncode != 0:
                self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Forecasting CLI error: {result.stderr}\n")
            else:
                # Log the CLI output
                if result.stdout.strip():
                    self.root.after(0, self.batch_log_text.insert, tk.END, f"         📊 {result.stdout.strip()}\n")
                # Show where forecasts are saved
                self.root.after(0, self.batch_log_text.insert, tk.END,
                    f"         📁 Forecast files saved to: {forecasts_dir}\n")
            return result.returncode == 0
        except Exception as e:
            self.root.after(0, self.batch_log_text.insert, tk.END, f"         ❌ Forecasting CLI error: {e}\n")
            return False
    
    def stop_batch_processing(self):
        """Stop the batch processing"""
        if self.batch_processing_active:
            self.batch_processing_active = False
            self.batch_status_var.set("Stopping...")
            self.batch_log_text.insert(tk.END, "\n⏹️ Batch processing stopped by user\n")
            messagebox.showinfo("Stopped", "Batch processing has been stopped.")
        else:
            messagebox.showinfo("Info", "No batch processing is currently running.")
    
    def open_batch_output(self):
        """Open the batch processing output folder"""
        output_dir = Path(self.batch_output_folder_var.get())
        if output_dir.exists():
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(output_dir)])
            else:
                subprocess.run(["xdg-open", str(output_dir)])
        else:
            messagebox.showwarning("Directory Not Found", f"Output directory does not exist: {output_dir}")
    
    def clear_batch_log(self):
        """Clear the batch processing log"""
        self.batch_log_text.delete(1.0, tk.END)
        self.batch_log_text.insert(tk.END, "🔄 Batch Processing Log Cleared\n")
        self.batch_log_text.insert(tk.END, "=" * 40 + "\n\n")
        self.auto_scroll_batch_log()
    
    def auto_scroll_batch_log(self):
        """Auto-scroll the batch log to the bottom and ensure canvas scroll region is updated"""
        try:
            # Scroll the log text to the bottom
            self.batch_log_text.see(tk.END)
            # Update the canvas scroll region to ensure proper scrolling
            if hasattr(self, 'batch_canvas'):
                self.batch_canvas.configure(scrollregion=self.batch_canvas.bbox("all"))
        except Exception:
            pass  # Silently handle any scrolling errors
    
    def log_batch_message(self, message):
        """Helper method to add a message to the batch log with auto-scrolling"""
        def _update_log():
            self.batch_log_text.insert(tk.END, message)
            self.auto_scroll_batch_log()
        
        if hasattr(self, 'root'):
            self.root.after(0, _update_log)
        else:
            _update_log()


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SiloFlowTester(root)
    
    # Start the application - logs tab removed as requested
    root.mainloop()

if __name__ == "__main__":
    import sys
    main()
