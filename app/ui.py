import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import os
import zipfile
import shutil
import json
import pandas as pd
from PIL import Image
from dataset_check import check_dataset

st.set_page_config(page_title="Food Classifier Dashboard", page_icon="🍔", layout="wide")

APPLE_STYLE_CSS = """
<style>
    /* Apple-like Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        font-weight: 400;
        color: #1d1d1f;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.015em;
        color: #1d1d1f;
    }
    
    /* Simplified Header/Title */
    .stApp > header {
        background-color: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px);
    }
    
    /* Main background */
    .stApp {
        background-color: #f5f5f7;
    }
    
    /* Style the tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e5e5ea;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 4px 16px;
        color: #1d1d1f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1d1d1f;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #007aff;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #006ee6;
        color: white;
        box-shadow: 0 4px 8px rgba(0, 122, 255, 0.3);
    }
    
    /* Text Inputs and Radio Buttons */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #d2d2d7;
        background-color: #ffffff;
    }
    .stRadio > div {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border: 1px solid #e5e5ea;
    }
    
    /* Expander / Dataframes / Containers */
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e5ea;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        overflow: hidden;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-weight: 600;
        color: #1d1d1f;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #86868b;
    }
    
    /* Info/Warning/Success Messages */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
</style>
"""

st.markdown(APPLE_STYLE_CSS, unsafe_allow_html=True)

st.title("Food Classifier Dashboard")

tab1, tab2 = st.tabs(["Dataset Uploader", "Training Results"])

with tab1:
    st.markdown("Upload your dataset zip file or enter a local path to your dataset.")
    upload_mode = st.radio("Choose Dataset Source", ["ZIP File Upload", "Local Directory Path"])
    
    search_dir = None
    ready_to_check = False
    
    if upload_mode == "ZIP File Upload":
        uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
        DATA_DIR = "uploaded_dataset"
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            if st.button("Extract and Check Dataset"):
                if os.path.exists(DATA_DIR):
                    shutil.rmtree(DATA_DIR)
                os.makedirs(DATA_DIR, exist_ok=True)
                
                zip_path = os.path.join(DATA_DIR, "dataset.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                with st.spinner("Extracting dataset..."):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(DATA_DIR)
                        
                st.success(f"Dataset extracted to `{DATA_DIR}`")
                os.remove(zip_path)
                
                search_dir = DATA_DIR
                extracted_items = os.listdir(DATA_DIR)
                if len(extracted_items) == 1 and os.path.isdir(os.path.join(DATA_DIR, extracted_items[0])):
                    search_dir = os.path.join(DATA_DIR, extracted_items[0])
                ready_to_check = True
    else:
        local_path = st.text_input("Enter the absolute path to your dataset folder:")
        if local_path and os.path.exists(local_path):
            st.success("Directory found!")
            if st.button("Check Dataset"):
                search_dir = local_path
                ready_to_check = True
        elif local_path:
            st.error("Directory not found. Please check the path.")
    
    if ready_to_check and search_dir:
        with st.spinner("Running dataset checks..."):
            report = check_dataset(search_dir)
            
        st.subheader("Dataset Report")
        st.metric("Total Images", report.get("total_images", 0))
        
        st.write("### Class Distribution")
        class_counts = report.get("class_counts", {})
        if class_counts:
            cols = st.columns(min(len(class_counts), 4) or 1)
            for i, (cls_name, count) in enumerate(class_counts.items()):
                cols[i % len(cols)].metric(f"{cls_name}", count)
        
        st.write("### Quality Metrics")
        col4, col5, col6 = st.columns(3)
        col4.metric("Corrupted Images", len(report.get("corrupted_images", [])))
        col5.metric("Unsupported Formats", len(report.get("unsupported_formats", [])))
        col6.metric("Duplicates (Groups)", len(report.get("near_duplicate_groups", [])))
        
        if report.get("class_imbalance_ratio"):
            st.info(f"Class Imbalance Ratio (Max/Min): {report['class_imbalance_ratio']:.2f}")
            
        with st.expander("View Full Detailed JSON Report"):
            st.json(report)

with tab2:
    st.header("Training & Evaluation Results")
    
    model_tabs = st.tabs(["MobileNet V3 Large", "EfficientNet B2"])
    models = ["mobilenet_v3_large", "efficientnet_b2"]
    
    for i, selected_model in enumerate(models):
        with model_tabs[i]:
            results_dir = f"results_{selected_model}"
            
            if not os.path.exists(f"{results_dir}/split_info.json"):
                st.warning(f"No training results found for {selected_model}. Please ensure `train.py` and `evaluate.py` have been run.")
                continue
                
            # Load Data
            with open(f"{results_dir}/split_info.json", "r") as f:
                split_info = json.load(f)
            
            report_path = f"{results_dir}/classification_report.json"
            clf_report = {}
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    clf_report = json.load(f)
        
            train_files = split_info.get("train_files", [])
            val_files = split_info.get("val_files", [])
            total_images = len(train_files) + len(val_files)
        
            # Dish Mapping for Breakdown
            dish_mapping = {
                "cheese_pie": "veg", "grilled_kofta": "non_veg", "hummus": "veg",
                "kofta_stew": "non_veg", "meat_maqluba": "non_veg", "meat_pie": "non_veg",
                "mixed_grill": "non_veg", "okra_tagine": "veg", "onion_soup": "veg", 
                "pistachio_sweet": "veg", "pistachio_tamriya": "veg", "pomegranate_fattoush": "veg",
                "rice_pudding": "veg", "saffron_cake": "veg", "sesame_brittle": "veg",
                "shawarma": "non_veg", "shrimp_sayadiyah": "non_veg", "spinach_fatayer": "veg",
                "spinach_stew": "veg", "stuffed_pigeon": "non_veg", "tamees_bread": "veg",
                "tamriya": "veg", "za’atar_manakish": "veg"
            }
        
            def get_breakdown(files):
                found_dishes = set()
                v_count, nv_count = 0, 0
                for f in files:
                    basename = os.path.basename(f)
                    for dish, category in dish_mapping.items():
                        if basename.startswith(dish + "_"):
                            if dish not in found_dishes:
                                found_dishes.add(dish)
                                if category == "veg": v_count += 1
                                else: nv_count += 1
                            break
                return f"{v_count} Veg, {nv_count} Non-Veg", len(found_dishes)

            train_bd, train_dish_count = get_breakdown(train_files)
            val_bd, val_dish_count = get_breakdown(val_files)
            total_bd, total_dish_count = get_breakdown(train_files + val_files)
            
            # 1. Dataset Split Breakdown
            st.markdown(f"###  Dataset Split Breakdown (for `20_dishes` - {total_images} Total Images)")
            split_data = [
                {"Split": " Train", "Target Ratio": "80%", "Actual Images": f"{len(train_files)} images", "Percentage": f"{(len(train_files)/total_images)*100:.1f}%", "Dish Count": f"{train_dish_count} dishes", "Dish Breakdown": train_bd},
                {"Split": "🔍 Validation", "Target Ratio": "20%", "Actual Images": f"{len(val_files)} images", "Percentage": f"{(len(val_files)/total_images)*100:.1f}%", "Dish Count": f"{val_dish_count} dishes", "Dish Breakdown": val_bd},
                {"Split": "Total", "Target Ratio": "100%", "Actual Images": f"{total_images} images", "Percentage": "100%", "Dish Count": f"{total_dish_count} dishes", "Dish Breakdown": total_bd}
            ]
            st.dataframe(pd.DataFrame(split_data), use_container_width=True, hide_index=True)
            
            # 2. Key Ratios
            st.markdown("### 📐 Key Ratios")
            train_val_ratio = len(train_files) / max(len(val_files), 1)
            st.markdown(f"- **Train-to-Validation Ratio:** {train_val_ratio:.2f} : 1 (~{int((len(train_files)/total_images)*100)}% Train vs ~{int((len(val_files)/total_images)*100)}% Validation).")
            
            st.divider()
            
            # 3. Class Counts Table
            st.markdown("### Total Class Counts")
            if os.path.exists("results/dataset_report.json"):
                with open("results/dataset_report.json", "r") as f:
                    ds_report = json.load(f)
                class_counts = ds_report.get("class_counts", {})
                cc_data = [
                    {"Category": "🥬 Vegetarian (Veg)", "Dish Count": "13 dishes", "Total Images": f"{class_counts.get('veg', 0)} images"},
                    {"Category": "🥩 Non-Vegetarian (Non-Veg)", "Dish Count": "10 dishes", "Total Images": f"{class_counts.get('non_veg', 0)} images"},
                    {"Category": "Total", "Dish Count": "23 dishes", "Total Images": f"{total_images} images"}
                ]
                st.dataframe(pd.DataFrame(cc_data), use_container_width=True, hide_index=True)
                
            st.divider()
            
            # 4. Per-Class Performance Breakdown
            if clf_report:
                st.markdown("### Per-Class Performance Breakdown:")
                
                veg_metrics = clf_report.get("veg", {})
                non_veg_metrics = clf_report.get("non_veg", {})
                
                def format_metric(v_val, nv_val, metric, is_pct=True, sup_v=None, sup_nv=None):
                    if is_pct:
                        return {"Metric": metric, "🥬 VEG (Class 0)": f"{v_val*100:.2f}%", "🥩 NON-VEG (Class 1)": f"{nv_val*100:.2f}%"}
                    return {"Metric": metric, "🥬 VEG (Class 0)": f"{int(v_val)} images", "🥩 NON-VEG (Class 1)": f"{int(nv_val)} images"}
                    
                perf_data = [
                    format_metric(veg_metrics.get("precision", 0), non_veg_metrics.get("precision", 0), "Precision"),
                    format_metric(veg_metrics.get("recall", 0), non_veg_metrics.get("recall", 0), "Recall"),
                    format_metric(veg_metrics.get("f1-score", 0), non_veg_metrics.get("f1-score", 0), "F1-Score"),
                    format_metric(veg_metrics.get("support", 0), non_veg_metrics.get("support", 0), "Support (True Images)", is_pct=False)
                ]
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
                
                st.divider()
                
                # 5. Validation Set Confusion Matrix
                st.markdown("### 📊 Validation Set Confusion Matrix")
                
                true_veg = round(veg_metrics.get("recall", 0) * veg_metrics.get("support", 0))
                false_non_veg = int(veg_metrics.get("support", 0)) - true_veg
                true_non_veg = round(non_veg_metrics.get("recall", 0) * non_veg_metrics.get("support", 0))
                false_veg = int(non_veg_metrics.get("support", 0)) - true_non_veg
                
                cm_data = [
                    {" ": "Actual VEG", "Predicted VEG": f"{true_veg} (True VEG)", "Predicted NON-VEG": f"{false_non_veg} (False NON-VEG)", "Total True Samples": int(veg_metrics.get("support", 0))},
                    {" ": "Actual NON-VEG", "Predicted VEG": f"{false_veg} (False VEG)", "Predicted NON-VEG": f"{true_non_veg} (True NON-VEG)", "Total True Samples": int(non_veg_metrics.get("support", 0))}
                ]
                st.dataframe(pd.DataFrame(cm_data), use_container_width=True, hide_index=True)
                
                # Validation Performance Metrics
                st.markdown("### 📈 Validation Performance Metrics")
                acc = clf_report.get("accuracy", 0)
                macro_f1 = clf_report.get("macro avg", {}).get("f1-score", 0)
                weighted_f1 = clf_report.get("weighted avg", {}).get("f1-score", 0)
                
                st.markdown(f"- **Validation Accuracy:** {acc*100:.2f}%")
                st.markdown(f"- **Macro F1-Score:** {macro_f1*100:.2f}%")
                st.markdown(f"- **Weighted F1-Score:** {weighted_f1*100:.2f}%")

