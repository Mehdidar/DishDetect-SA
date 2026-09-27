import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re

with open("ui.py", "r") as f:
    content = f.read()

# Find the start of tab2
tab2_start = content.find("with tab2:")

if tab2_start != -1:
    content = content[:tab2_start]
    
new_tab2 = """with tab2:
    st.header("Training & Evaluation Results")
    
    if not os.path.exists("results/split_info.json"):
        st.warning("No training results found. Please ensure `train.py` and `evaluate.py` have been run.")
    else:
        # Load Data
        with open("results/split_info.json", "r") as f:
            split_info = json.load(f)
        
        report_path = "results/classification_report.json"
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
        st.markdown(f"### 📊 Dataset Split Breakdown (for `20_dishes` - {total_images} Total Images)")
        split_data = [
            {"Split": "🏋️ Train", "Target Ratio": "70%", "Actual Images": f"{len(train_files)} images", "Percentage": f"{(len(train_files)/total_images)*100:.1f}%", "Dish Count": f"{train_dish_count} dishes", "Dish Breakdown": train_bd},
            {"Split": "🔍 Validation", "Target Ratio": "30%", "Actual Images": f"{len(val_files)} images", "Percentage": f"{(len(val_files)/total_images)*100:.1f}%", "Dish Count": f"{val_dish_count} dishes", "Dish Breakdown": val_bd},
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
            
            def format_metric(v_val, nv_val, metric, is_pct=True):
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
                {"": "Actual VEG", "Predicted VEG": f"{true_veg} (True VEG)", "Predicted NON-VEG": f"{false_non_veg} (False NON-VEG)", "Total True Samples": int(veg_metrics.get("support", 0))},
                {"": "Actual NON-VEG", "Predicted VEG": f"{false_veg} (False VEG)", "Predicted NON-VEG": f"{true_non_veg} (True NON-VEG)", "Total True Samples": int(non_veg_metrics.get("support", 0))}
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
"""

    with open("ui.py", "w") as f:
        f.write(content + new_tab2)
