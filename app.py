import av
import cv2
import numpy as np
import streamlit as st
import os
from collections import Counter
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import tempfile
import time # Adicionado para garantir o FPS
import torch
from ultralytics.nn.tasks import DetectionModel 
# Configuração do Streamlit
st.set_page_config(page_title="Contador de Objetos", layout="wide")
st.title("📷 Contador de Objetos — Streamlit com YOLOv8")

# --- Variáveis de Estado para o WebRTC ---
# Se o estado do slider mudar, o Streamlit roda o script novamente.
# Vamos armazenar os valores de modo que o webrtc_streamer possa ser recriado com novos valores.

# --- Verificação e Carregamento de Modelos ---
MODEL_DIR = "models"
yolo_model = None

st.sidebar.header("Configurações")
mode = st.sidebar.selectbox("Modo de contagem", ["YOLO (Deep Learning)", "Clássico (Hough - círculos)"])

# ---------- Parâmetros comuns ----------
draw_boxes = st.sidebar.checkbox("Desenhar anotações", True)
show_fps = st.sidebar.checkbox("Mostrar FPS", True)

# === Seleção da Fonte de Mídia e Upload de Arquivos ===
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
conf_thres = 0.4 # Default
iou_thres = 0.5  # Default
max_det = 300    # Default

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

        # Atualizando os sliders (agora com chaves/keys para forçar a atualização)
        conf_thres = st.sidebar.slider("Confiança mínima", 0.1, 0.9, 0.4, 0.05, key="conf_slider")
        iou_thres = st.sidebar.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05, key="iou_slider")
        max_det = st.sidebar.slider("Máximo de detecções", 50, 1000, 300, 10, key="maxdet_slider")

        @st.cache_resource(show_spinner="Carregando Modelo YOLO...")
        def load_model(path):
            # Envolve a chamada YOLO com o gerenciador de contexto de segurança do PyTorch
            with torch.serialization.safe_globals([DetectionModel]):
                return YOLO(path)
        
        yolo_model = load_model(model_path)

    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        st.stop()

# ---------- Lógica Hough (Clássico) ----------
else:
    dp = st.sidebar.slider("dp (resolução)", 0.8, 3.0, 1.2, 0.1)
    minDist = st.sidebar.slider("minDist entre centros", 5, 200, 28, 1)
    canny = st.sidebar.slider("Canny (param1)", 50, 300, 120, 1)
    acc_thr = st.sidebar.slider("Acumulador (param2)", 10, 200, 35, 1)
    minR = st.sidebar.slider("Raio mínimo", 0, 200, 12, 1)
    maxR = st.sidebar.slider("Raio máximo", 0, 400, 60, 1)


# ---------- Configuração WebRTC Simplificada ----------
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# =========================================================================
# === CLASSE TRANSFORMER (Lógica de Processamento) ===
# =========================================================================
class ObjectCounterTransformer(VideoTransformerBase):
    def __init__(self, yolo_model=None, conf_thres=0.4, iou_thres=0.5, max_det=300): # Parâmetros adicionados
        self.yolo_model = yolo_model
        self.conf_thres = conf_thres # Armazena o valor do slider
        self.iou_thres = iou_thres   # Armazena o valor do slider
        self.max_det = max_det       # Armazena o valor do slider
        self.counter = Counter()
        self.fps_hist = []
        self.prev_time = cv2.getTickCount() / cv2.getTickFrequency()

    def _update_fps(self):
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        self.fps_hist.append(fps)
        if len(self.fps_hist) > 30:
            self.fps_hist.pop(0)
        return np.mean(self.fps_hist)
    
    def process_frame(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        self.counter = Counter()

        try:
            if mode.startswith("YOLO") and self.yolo_model is not None:
                # Processamento YOLO (AGORA USANDO self.VARIAVEL)
                results = self.yolo_model.predict(
                    source=img,
                    conf=self.conf_thres, # Lendo o valor armazenado no __init__
                    iou=self.iou_thres,   # Lendo o valor armazenado no __init__
                    verbose=False,
                    max_det=self.max_det
                )
                
                detections = results[0]
                
                if detections.boxes is not None and len(detections.boxes) > 0:
                    boxes = detections.boxes.xyxy.cpu().numpy().astype(int)
                    classes = detections.boxes.cls.cpu().numpy().astype(int)
                    class_names = detections.names
                    
                    for (x1, y1, x2, y2), class_id in zip(boxes, classes):
                        class_name = class_names.get(class_id, str(class_id))
                        self.counter[class_name] += 1
                        
                        if draw_boxes:
                            # Desenha caixa (Cor Verde)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Desenha nome (Cor Verde)
                            cv2.putText(img, class_name, (x1, max(10, y1-6)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # --- Exibir Contadores (HUD) ---
            y_offset = 30
            for obj_type, count in sorted(self.counter.items()):
                text = f"{obj_type}: {count}"
                # Desenha o texto da contagem (Cor Vermelha para garantir visibilidade)
                cv2.putText(img, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 30

        except Exception as e:
            error_text = f"Erro no YOLO: {str(e)}"
            cv2.putText(img, error_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return img
    
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        try:
            if frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            img = frame.to_ndarray(format="bgr24")
            processed_img = self.process_frame(img)
            
            if show_fps:
                fps = self._update_fps()
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

if source_option == "Webcam (Live Stream)":
    st.info("🔴 **Webcam ao Vivo** - Clique em 'START' para iniciar a câmera. **ATENÇÃO: Após ajustar os sliders, clique em STOP e START novamente!**")
    
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
        
        # 🚨 PASSANDO PARÂMETROS ATUAIS PARA A FÁBRICA
        video_processor_factory=lambda: ObjectCounterTransformer(
            yolo_model=yolo_model,
            conf_thres=conf_thres, # Valor atual do slider
            iou_thres=iou_thres,   # Valor atual do slider
            max_det=max_det        # Valor atual do slider
        ),
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("✅ Câmera ativa - Detecção em andamento")
    else:
        st.warning("⏸️ Clique em START para iniciar a câmera")

elif uploaded_file is not None:
    if source_option == "Carregar Imagem da Galeria":
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # 🚨 CORREÇÃO ROBUSTA DE ERRO DE IMAGEM
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                 st.error("❌ Erro: O OpenCV não conseguiu ler a imagem. Tente salvar o arquivo em outro formato (ex: PNG) ou use uma imagem diferente.")
                 # O return aqui evita o erro NoneType no processamento
                 st.stop()
            
            transformer = ObjectCounterTransformer(
                yolo_model=yolo_model, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres,
                max_det=max_det
            )
            processed_img = transformer.process_frame(img)
            
            st.image(processed_img, channels="BGR", 
                     caption="🔍 Resultado da Detecção", 
                     use_column_width=True)
            
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
            transformer = ObjectCounterTransformer(yolo_model=yolo_model, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
            
            stframe = st.empty()
            stop_processing = st.button("⏹️ Parar Processamento")
            
            if cap.isOpened():
                st.info("🎥 Processando vídeo...")
                
                while cap.isOpened() and not stop_processing:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame = transformer.process_frame(frame)
                    stframe.image(processed_frame, channels="BGR", 
                                  use_column_width=True)
                    
                    # Atualizar contagem
                    if transformer.counter:
                        count_text = " | ".join([f"**{k}**: {v}" for k, v in transformer.counter.items()])
                        # Mostra a contagem fora da imagem no modo vídeo
                        st.info(f"📊 **Contagem Atual:** {count_text}") 
                    
                cap.release()
                
                if not stop_processing:
                    st.success("✅ Processamento do vídeo concluído!")
                    final_count = " | ".join([f"**{k}**: {v}" for k, v in transformer.counter.items()])
                    st.info(f"📋 **Contagem Final:** {final_count}")
                else:
                    st.warning("⏹️ Processamento interrompido pelo usuário")
                    
            else:
                st.error("❌ Erro: Não foi possível abrir o vídeo")
                
        except Exception as e:
            st.error(f"❌ Erro no processamento do vídeo: {str(e)}")
        finally:
            if 'tfile' in locals():
                os.unlink(tfile.name)

else:
    if source_option != "Webcam (Live Stream)":
        st.info("📁 Por favor, carregue um arquivo na barra lateral para iniciar o processamento.")


# =========================================================================
# === INSTRUÇÕES DE USO ===
# =========================================================================
st.markdown("---")
st.markdown("""
### 🎯 **Instruções de Uso:**

**🔴 Webcam:**
- Clique em **START** para iniciar a câmera
- **IMPORTANTE**: Após ajustar a **Confiança** ou **IoU NMS**, clique em **STOP** e depois em **START** novamente. Isso recria a conexão da câmera com os novos parâmetros.

**⚙️ Configurações:**
- **IoU NMS**: **Reduza este valor** (ex: para **0.30** ou **0.15**) para contar objetos que estão tocando ou empilhados.
""")