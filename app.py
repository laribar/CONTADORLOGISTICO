import av
import cv2
import numpy as np
import streamlit as st
import os
from collections import Counter
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import tempfile
import time
import torch
from ultralytics.nn.tasks import DetectionModel 
from torch.nn.modules.container import Sequential

# Configuração do Streamlit
st.set_page_config(page_title="Contador de Objetos", layout="wide")
st.title("📷 Contador de Objetos — Streamlit com YOLOv8")

# --- Variáveis de Estado ---
if 'processing' not in st.session_state:
    st.session_state.processing = False

# --- Verificação e Carregamento de Modelos ---
MODEL_DIR = "models"
yolo_model = None

st.sidebar.header("Configurações")
mode = st.sidebar.selectbox("Modo de contagem", ["YOLO (Deep Learning)", "Clássico (Hough - círculos)"])

# ---------- Parâmetros comuns ----------
draw_boxes = st.sidebar.checkbox("Desenhar anotações", True)
show_fps = st.sidebar.checkbox("Mostrar FPS", True)

# === Seleção da Fonte de Mídia ===
source_option = st.sidebar.selectbox(
    "Escolher Fonte de Mídia",
    ["Webcam (Live Stream)", "Carregar Imagem da Galeria", "Carregar Vídeo Local"]
)

uploaded_file = None
if source_option == "Carregar Imagem da Galeria":
    uploaded_file = st.sidebar.file_uploader("Carregue um arquivo de imagem (PNG, JPG)", type=["png", "jpg", "jpeg"])
elif source_option == "Carregar Vídeo Local":
    uploaded_file = st.sidebar.file_uploader("Carregue um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

# ---------- Lógica YOLO (Deep Learning) ----------
conf_thres = 0.4
iou_thres = 0.5
max_det = 300

if mode.startswith("YOLO"):
    try:
        if not os.path.exists(MODEL_DIR):
            st.error(f"❌ Erro: Pasta '{MODEL_DIR}' não encontrada. Crie-a e adicione seus modelos .pt.")
            st.stop()

        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]

        if not model_files:
            st.error(f"❌ Erro: Nenhum arquivo .pt encontrado em '{MODEL_DIR}'.")
            st.stop()

        selected_model_file = st.sidebar.selectbox("Escolher Modelo YOLO", model_files)
        model_path = os.path.join(MODEL_DIR, selected_model_file)

        # Sliders para parâmetros YOLO
        conf_thres = st.sidebar.slider("Confiança mínima", 0.1, 0.9, 0.4, 0.05, key="conf_slider")
        iou_thres = st.sidebar.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05, key="iou_slider")
        max_det = st.sidebar.slider("Máximo de detecções", 50, 1000, 300, 10, key="maxdet_slider")

        @st.cache_resource(show_spinner="Carregando Modelo YOLO...")
        def load_model(path):
            try:
                # Primeira tentativa: carregamento normal
                st.info("🔄 Tentando carregamento normal...")
                return YOLO(path)
            except Exception as e:
                st.warning(f"⚠️ Carregamento normal falhou: {e}")
                st.info("🔄 Tentando carregar com safe_globals...")
                try:
                    # Segunda tentativa: com safe_globals para ambas as classes
                    with torch.serialization.safe_globals([DetectionModel, Sequential]):
                        model = YOLO(path)
                        st.success("✅ Modelo carregado com safe_globals!")
                        return model
                except Exception as e2:
                    st.error(f"❌ Erro também com safe_globals: {e2}")
                    st.info("🔄 Tentando método alternativo...")
                    try:
                        # Terceira tentativa: carregamento direto com torch.load
                        weights = torch.load(path, weights_only=False)
                        model = YOLO(path)
                        st.success("✅ Modelo carregado com método alternativo!")
                        return model
                    except Exception as e3:
                        st.error(f"❌ Todas as tentativas falharam: {e3}")
                        st.stop()
        
        yolo_model = load_model(model_path)
        st.sidebar.success(f"✅ Modelo {selected_model_file} carregado!")

    except Exception as e:
        st.error(f"❌ Erro crítico ao carregar o modelo: {e}")
        st.stop()

# ---------- Lógica Hough (Clássico) ----------
else:
    dp = st.sidebar.slider("dp (resolução)", 0.8, 3.0, 1.2, 0.1)
    minDist = st.sidebar.slider("minDist entre centros", 5, 200, 28, 1)
    canny = st.sidebar.slider("Canny (param1)", 50, 300, 120, 1)
    acc_thr = st.sidebar.slider("Acumulador (param2)", 10, 200, 35, 1)
    minR = st.sidebar.slider("Raio mínimo", 0, 200, 12, 1)
    maxR = st.sidebar.slider("Raio máximo", 0, 400, 60, 1)

# ---------- Configuração WebRTC ----------
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# =========================================================================
# === CLASSE TRANSFORMER (Lógica de Processamento) ===
# =========================================================================
class ObjectCounterTransformer(VideoTransformerBase):
    def __init__(self, yolo_model=None, conf_thres=0.4, iou_thres=0.5, max_det=300):
        self.yolo_model = yolo_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.counter = Counter()
        self.fps_hist = []
        self.prev_time = time.time()
        self.frame_count = 0

    def _update_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        self.fps_hist.append(fps)
        if len(self.fps_hist) > 30:
            self.fps_hist.pop(0)
        return np.mean(self.fps_hist) if self.fps_hist else 0
    
    def process_frame(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        self.counter = Counter()
        output_img = img.copy()

        try:
            if mode.startswith("YOLO") and self.yolo_model is not None:
                # Processamento YOLO
                results = self.yolo_model.predict(
                    source=img,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    verbose=False,
                    max_det=self.max_det
                )
                
                detections = results[0]
                
                if detections.boxes is not None and len(detections.boxes) > 0:
                    boxes = detections.boxes.xyxy.cpu().numpy().astype(int)
                    classes = detections.boxes.cls.cpu().numpy().astype(int)
                    confidences = detections.boxes.conf.cpu().numpy()
                    class_names = detections.names
                    
                    for (x1, y1, x2, y2), class_id, conf in zip(boxes, classes, confidences):
                        class_name = class_names.get(class_id, str(class_id))
                        self.counter[class_name] += 1
                        
                        if draw_boxes:
                            # Desenha caixa (Cor Verde)
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Label com confiança
                            label = f"{class_name} {conf:.2f}"
                            # Desenha fundo para o texto
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            cv2.rectangle(output_img, (x1, y1 - text_height - 10), 
                                        (x1 + text_width, y1), (0, 255, 0), -1)
                            # Desenha texto
                            cv2.putText(output_img, label, (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # --- Exibir Contadores (HUD) ---
            y_offset = 35
            for obj_type, count in sorted(self.counter.items()):
                text = f"{obj_type}: {count}"
                # Desenha fundo para contador
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(output_img, (5, y_offset - text_height - 5), 
                            (15 + text_width, y_offset + 5), (0, 0, 0), -1)
                # Desenha texto da contagem
                cv2.putText(output_img, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += 35

        except Exception as e:
            error_text = f"Erro no processamento: {str(e)}"
            cv2.putText(output_img, error_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return output_img
    
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        try:
            if frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            img = frame.to_ndarray(format="bgr24")
            processed_img = self.process_frame(img)
            
            if show_fps:
                fps = self._update_fps()
                # Fundo para FPS
                cv2.rectangle(processed_img, (img.shape[1] - 130, 5), 
                            (img.shape[1] - 5, 40), (0, 0, 0), -1)
                # Texto FPS
                cv2.putText(processed_img, f"FPS: {fps:.1f}", 
                            (img.shape[1] - 120, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            return processed_img
            
        except Exception as e:
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Erro no Transform: {str(e)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return error_img

# =========================================================================
# === LÓGICA PRINCIPAL DE EXIBIÇÃO ===
# =========================================================================

def stop_processing():
    st.session_state.processing = False

if source_option == "Webcam (Live Stream)":
    st.info("🔴 **Webcam ao Vivo** - Clique em 'START' para iniciar a câmera.")
    st.warning("⚠️ **ATENÇÃO**: Após ajustar os sliders, clique em STOP e START novamente!")
    
    webrtc_ctx = webrtc_streamer(
        key="object-counter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280, "min": 640}, 
                "height": {"ideal": 720, "min": 480},
                "frameRate": {"ideal": 20} 
            }, 
            "audio": False
        },
        video_processor_factory=lambda: ObjectCounterTransformer(
            yolo_model=yolo_model,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        ),
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("✅ Câmera ativa - Detecção em andamento")
    else:
        st.info("⏸️ Clique em START para iniciar a câmera")

elif uploaded_file is not None:
    if source_option == "Carregar Imagem da Galeria":
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("❌ Erro: Não foi possível decodificar a imagem.")
                st.stop()
            
            # Redimensionar se muito grande
            height, width = img.shape[:2]
            if width > 1200 or height > 800:
                scale = min(1200/width, 800/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            transformer = ObjectCounterTransformer(
                yolo_model=yolo_model, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres,
                max_det=max_det
            )
            processed_img = transformer.process_frame(img)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, channels="BGR", caption="📷 Imagem Original", use_column_width=True)
            with col2:
                st.image(processed_img, channels="BGR", caption="🔍 Resultado da Detecção", use_column_width=True)
            
            if transformer.counter:
                count_text = " | ".join([f"**{k}**: {v}" for k, v in transformer.counter.items()])
                st.success(f"📊 **Contagem Total:** {count_text}")
            else:
                st.warning("⚠️ Nenhum objeto detectado")
                
        except Exception as e:
            st.error(f"❌ Erro ao processar imagem: {str(e)}")

    elif source_option == "Carregar Vídeo Local":
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush()
            
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("❌ Erro: Não foi possível abrir o vídeo.")
                st.stop()
            
            transformer = ObjectCounterTransformer(
                yolo_model=yolo_model, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres, 
                max_det=max_det
            )
            
            stframe = st.empty()
            stop_button = st.button("⏹️ Parar Processamento", on_click=stop_processing)
            
            st.session_state.processing = True
            st.info("🎥 Processando vídeo...")
            
            while cap.isOpened() and st.session_state.processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = transformer.process_frame(frame)
                stframe.image(processed_frame, channels="BGR", use_column_width=True)
                
                # Atualizar contagem
                if transformer.counter:
                    count_text = " | ".join([f"**{k}**: {v}" for k, v in transformer.counter.items()])
                    st.info(f"📊 **Contagem Atual:** {count_text}")
                
                time.sleep(0.03)  # Controlar velocidade
            
            cap.release()
            
            if not st.session_state.processing:
                st.warning("⏹️ Processamento interrompido pelo usuário")
            else:
                st.success("✅ Processamento do vídeo concluído!")
                final_count = " | ".join([f"**{k}**: {v}" for k, v in transformer.counter.items()])
                st.info(f"📋 **Contagem Final:** {final_count}")
                
        except Exception as e:
            st.error(f"❌ Erro no processamento do vídeo: {str(e)}")
        finally:
            if 'tfile' in locals():
                try:
                    os.unlink(tfile.name)
                except:
                    pass

else:
    if source_option != "Webcam (Live Stream)":
        st.info("📁 Por favor, carregue um arquivo na barra lateral para iniciar o processamento.")

# =========================================================================
# === SOLUÇÃO ALTERNATIVA SE AINDA NÃO FUNCIONAR ===
# =========================================================================
st.markdown("---")
st.markdown("### 🔧 Se o modelo ainda não carregar:")

st.code("""
# Solução alternativa manual no terminal:
pip install torch==2.5.1 --force-reinstall
""")

st.markdown("""
### 🎯 **Instruções de Uso:**

**🔴 Webcam:**
- Clique em **START** para iniciar a câmera
- **IMPORTANTE**: Após ajustar parâmetros, clique em **STOP** e depois **START**

**⚙️ Configurações YOLO:**
- **IoU NMS**: **Reduza para 0.15-0.30** para objetos empilhados
- **Confiança**: Ajuste conforme necessidade
""")