import os
import datetime
import cv2

class SessionLogger:
    def __init__(self, config, session_start_dt):
        self.config = config
        self.session_folder = ""
        self.images_folder = ""
        self.session_start_dt = session_start_dt
        
        logging_cfg = self.config.get("analytics_and_logging", {})
        model_cfg = self.config.get("model_variables", {})
        
        if logging_cfg.get("compare_responses", True):
            session_str = self.session_start_dt.strftime("%d-%m-%Y-%H-%M-%S")
            self.session_folder = os.path.join("logs", session_str)
            os.makedirs(self.session_folder, exist_ok=True)
            
            if logging_cfg.get("log_images", True):
                self.images_folder = os.path.join(self.session_folder, "images")
                os.makedirs(self.images_folder, exist_ok=True)
            
            if logging_cfg.get("log_token_usage", True):
                with open(os.path.join(self.session_folder, "token_usage.txt"), "w") as f:
                    f.write(f"{model_cfg.get('max_context', 131072)}\n")

    def log_session_metrics(
        self,
        step_type="inference",
        image_frame=None,
        timestamp_str="",
        inference_duration=None,
        total_tokens=None,
        messages_history=None,
        comparison_data=None
    ):
        logging_cfg = self.config.get("analytics_and_logging", {})
        
        if not logging_cfg.get("compare_responses", True) or not self.session_folder:
            return

        # 1. Log Black & White Frame Before Inference
        if step_type == "pre_inference" and logging_cfg.get("log_images", True) and image_frame is not None:
            gray_timeline = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            safe_time = timestamp_str.replace(":", "-")
            img_path = os.path.join(self.images_folder, f"frame_{safe_time}.jpg")
            cv2.imwrite(img_path, gray_timeline)

        # 2. Log Post-Inference Metrics (Execution times, tokens, history, and summaries)
        elif step_type == "inference":
            if logging_cfg.get("log_inference_times", True) and inference_duration is not None:
                with open(os.path.join(self.session_folder, "inference_times.txt"), "a") as f:
                    f.write(f"{inference_duration:.2f}\n")

            if logging_cfg.get("log_token_usage", True) and total_tokens is not None:
                with open(os.path.join(self.session_folder, "token_usage.txt"), "a") as f:
                    f.write(f"{total_tokens}\n")

            if logging_cfg.get("log_comparisons", True) and comparison_data:
                with open(os.path.join(self.session_folder, "comparison_logs.txt"), "a") as f:
                    f.write(f"Main Model message 1: {comparison_data.get('last_msg')}\n")
                    f.write(f"Main Model message 2: {comparison_data.get('current_msg')}\n")
                    f.write(f"Semantic score: {comparison_data.get('score')}\n")
                    f.write(f"Summary Model message: {comparison_data.get('summary_msg')}\n")
                    f.write("-" * 78 + "\n\n")

            if logging_cfg.get("log_transcript", True) and messages_history:
                with open(os.path.join(self.session_folder, "model_transcript.txt"), "w") as f:
                    for msg in messages_history:
                        f.write(f"[{msg['role'].capitalize()}]: {msg.get('content', '')}\n\n")

        # 3. Log Session Session Duration on Shutdown
        elif step_type == "runtime_end":
            session_end_dt = datetime.datetime.now()
            total_time = session_end_dt - self.session_start_dt
            
            tot_sec = int(total_time.total_seconds())
            hours = tot_sec // 3600
            minutes = (tot_sec % 3600) // 60
            seconds = tot_sec % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            with open(os.path.join(self.session_folder, "total_runtime.txt"), "w") as f:
                f.write(f"{self.session_start_dt.strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"{session_end_dt.strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"{time_str}\n")
