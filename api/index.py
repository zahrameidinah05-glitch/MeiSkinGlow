
import os
import sys

# ── Suppress TensorFlow & oneDNN verbose logs ──────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
import time
import numpy as np
import cv2
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, url_for
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))
UPLOAD_DIR      = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_SKIN_PATH = os.path.join(BASE_DIR, 'models', 'skin_type_model.tflite')
MODEL_ACNE_PATH = os.path.join(BASE_DIR, 'models', 'acne_type_model.tflite')

app.config['UPLOAD_FOLDER']      = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB

# No need to create UPLOAD_DIR on Vercel as we use in-memory processing

# ── Lazy-load CNN models ───────────────────────────────────────────────────────
model_skin = None
model_acne = None

try:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

def load_tflite_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def load_cnn_models():
    global model_skin, model_acne
    if model_skin is not None and model_acne is not None:
        return
    try:
        # Load Skin Type Model
        if os.path.exists(MODEL_SKIN_PATH):
            model_skin = load_tflite_model(MODEL_SKIN_PATH)
            print(f"[INFO] Successfully loaded Skin Type model from {MODEL_SKIN_PATH}")
        else:
            print(f"[WARN] Skin Type model not found at {MODEL_SKIN_PATH}")
            
        # Load Acne Type Model
        if os.path.exists(MODEL_ACNE_PATH):
            model_acne = load_tflite_model(MODEL_ACNE_PATH)
            print(f"[INFO] Successfully loaded Acne Type model from {MODEL_ACNE_PATH}")
        else:
            print(f"[WARN] Acne Type model not found at {MODEL_ACNE_PATH}")
            
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

def predict_tflite(interpreter, inp):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ── Class mapping (alphabetical matches) ──────────────────────────────────────
import unicodedata

def get_emoji(name_key):
    emoji_mapping = {
        'dry': 'DESERT',
        'normal': 'SLIGHTLY SMILING FACE',
        'oily': 'SPARKLES',
        'sensitive': 'CHERRY BLOSSOM',
        'none': 'GLOWING STAR',
        'blackheads': 'NOSE',
        'whiteheads': 'MEDIUM WHITE CIRCLE',
        'papules': 'LARGE RED CIRCLE',
        'pustules': 'LARGE YELLOW CIRCLE',
        'cysts': 'VOLCANO'
    }
    name = emoji_mapping.get(name_key, 'SLIGHTLY SMILING FACE')
    try:
        return unicodedata.lookup(name)
    except KeyError:
        return ""

SKIN_TYPE_CLASSES = ['dry', 'normal', 'oily', 'sensitive']
ACNE_TYPE_CLASSES = ['blackheads', 'cysts', 'none', 'papules', 'pustules', 'whiteheads']


# ── Skincare recommendation database ────────────────────────────────────────
SKIN_RECOMMENDATIONS = {
    'dry': {
        'name': 'Kulit Kering (Dry Skin)',
        'emoji': get_emoji('dry'),
        'description': 'Kulit wajah Anda cenderung kekurangan kelembapan dan minyak alami, sehingga terasa kencang, kasar, atau bersisik.',
        'recommended_ingredients': [
            'Hyaluronic Acid — Menghidrasi kulit secara mendalam.',
            'Ceramide — Memperbaiki dan menjaga skin barrier.',
            'Glycerin — Menarik kelembapan ke dalam sel kulit.',
            'Shea Butter — Pelembap alami yang mengunci hidrasi.'
        ],
        'avoided_ingredients': [
            'Alkohol Denat — Membuat kulit semakin kering dan dehidrasi.',
            'Salicylic Acid (BHA) konsentrasi tinggi — Dapat mengikis minyak alami kulit.',
            'Clay Mask — Menyerap minyak berlebih yang sebenarnya sangat dibutuhkan kulit kering.'
        ],
        'products': [
            'Cosrx Hyaluronic Acid Intensive Cream',
            'Skintific 5X Ceramide Barrier Moisture Gel',
            'The Lab by Blanc Doux Oligo Hyaluronic Acid Toner',
            'Cerave Hydrating Facial Cleanser'
        ]
    },
    'normal': {
        'name': 'Kulit Normal',
        'emoji': get_emoji('normal'),
        'description': 'Kulit wajah Anda memiliki keseimbangan yang baik antara kelembapan dan produksi sebum. Pori-pori tidak terlalu tampak dan jarang mengalami masalah serius.',
        'recommended_ingredients': [
            'Hyaluronic Acid — Menjaga hidrasi kulit agar tetap kenyal.',
            'Niacinamide — Menjaga kecerahan dan kekuatan skin barrier.',
            'Centella Asiatica — Menenangkan dan menyegarkan kulit wajah.',
            'Vitamin C — Memberikan perlindungan antioksidan dan mencerahkan wajah.'
        ],
        'avoided_ingredients': [
            'Eksfoliasi fisik berlebih — Bisa merusak keseimbangan alami kulit.',
            'Produk yang terlalu keras — Dapat merusak skin barrier yang sudah sehat.'
        ],
        'products': [
            'Simple Kind to Skin Hydrating Light Moisturizer',
            'Somethinc Low pH Gentle Jelly Cleanser',
            'Skintific 5X Ceramide Serum Sunscreen SPF50',
            'Avoskin Hydrating Treatment Essence'
        ]
    },
    'oily': {
        'name': 'Kulit Berminyak (Oily Skin)',
        'emoji': get_emoji('oily'),
        'description': 'Kulit wajah Anda memproduksi minyak berlebih (sebum), terutama di T-zone. Hal ini membuat wajah tampak mengilap dan rentan terhadap pori-pori tersumbat.',
        'recommended_ingredients': [
            'Salicylic Acid (BHA) — Membersihkan minyak di dalam pori-pori.',
            'Niacinamide — Mengontrol produksi sebum dan menyamarkan pori.',
            'Clay (Kaolin/Bentolit) — Menyerap kelebihan minyak wajah.',
            'Tea Tree Oil — Mengontrol minyak dan mencegah bakteri.'
        ],
        'avoided_ingredients': [
            'Heavy Oils (Coconut Oil, Mineral Oil) — Bersifat sangat komedogenik.',
            'Pelembap bertekstur krim tebal — Dapat menyumbat pori-pori wajah.',
            'Pembersih wajah yang terlalu mengikis — Memicu produksi sebum lebih banyak.'
        ],
        'products': [
            'Cosrx Salicylic Acid Daily Gentle Cleanser',
            'The Ordinary Niacinamide 10% + Zinc 1%',
            'Skintific Mugwort Clay Mask',
            'Somethinc Ceramic Skin Saviour Moisturizer Gel'
        ]
    },
    'sensitive': {
        'name': 'Kulit Sensitif (Sensitive Skin)',
        'emoji': get_emoji('sensitive'),
        'description': 'Kulit wajah Anda mudah mengalami kemerahan, perih, gatal, atau iritasi saat terpapar produk skincare baru atau faktor lingkungan.',
        'recommended_ingredients': [
            'Centella Asiatica (Cica) — Menenangkan kemerahan dan iritasi.',
            'Allantoin — Membantu penyembuhan jaringan kulit.',
            'Ceramide — Memperkuat barrier kulit agar tidak mudah terirritasi.',
            'Panthenol (Vitamin B5) — Menghidrasi sekaligus menenangkan kulit.'
        ],
        'avoided_ingredients': [
            'Pewangi Buatan (Fragrance/Parfum) — Pemicu utama iritasi kulit sensitif.',
            'Alkohol Denat & Essential Oils — Dapat menimbulkan efek terbakar atau kemerahan.',
            'Eksfoliator fisik kasar (Scrub) — Merusak skin barrier sensitif.'
        ],
        'products': [
            'Skintific 5X Ceramide Soothing Toner',
            'La Roche-Posay Cicaplast Baume B5',
            'Cetaphil Gentle Skin Cleanser',
            'Sensatia Botanicals Cleopatra\'s Rose Facial Hydrate'
        ]
    }
}

ACNE_RECOMMENDATIONS = {
    'none': {
        'name': 'Bebas Jerawat (Clean Skin)',
        'emoji': get_emoji('none'),
        'description': 'Tidak terdeteksi adanya jerawat aktif yang signifikan pada wajah Anda. Kulit tampak bersih dan sehat.',
        'recommended_ingredients': [],
        'avoided_ingredients': [],
        'products': []
    },
    'blackheads': {
        'name': 'Komedo Hitam (Blackheads)',
        'emoji': get_emoji('blackheads'),
        'description': 'Terdeteksi adanya bintik-bintik hitam di area pori-pori kulit Anda (Blackheads). Kondisi ini terjadi ketika sebum dan sel kulit mati menyumbat pori-pori dan teroksidasi saat terpapar udara.',
        'recommended_ingredients': [
            'Salicylic Acid (BHA) — Sangat efektif melarutkan komedo penyumbat pori.',
            'Glycolic Acid (AHA) — Mengeksfoliasi permukaan kulit untuk mengangkat sel kulit mati.',
            'Clay Mask (Kaolin/Bentolit) — Menyerap kelebihan minyak dan kotoran penyumbat.',
            'Retinol — Mempercepat pergantian sel kulit agar pori tidak mudah tersumbat.'
        ],
        'avoided_ingredients': [
            'Heavy Moisturizers — Pelembap tebal yang menyumbat pori.',
            'Pore Strips Kasar — Dapat merusak jaringan kulit dan memperbesar pori secara permanen.',
            'Lanolin & Isopropyl Myristate — Bahan pelembap komedogenik tinggi.'
        ],
        'products': [
            'Cosrx BHA Blackhead Power Liquid',
            'Somethinc AHA BHA PHA Peeling Solution',
            'The Ordinary Salicylic Acid 2% Solution',
            'Avoskin Miraculous Refining Toner'
        ]
    },
    'whiteheads': {
        'name': 'Komedo Putih (Whiteheads)',
        'emoji': get_emoji('whiteheads'),
        'description': 'Terdeteksi adanya bintik putih kecil atau tekstur kulit kasar/bergerindil (Whiteheads). Kondisi ini terjadi ketika sel kulit mati dan sebum menyumbat pori-pori namun tetap tertutup oleh lapisan tipis kulit.',
        'recommended_ingredients': [
            'Salicylic Acid (BHA) — Membantu mengeksfoliasi di dalam pori-pori tersumbat.',
            'Glycolic Acid (AHA) — Membantu mengangkat sel kulit mati dan menghaluskan tekstur kasar.',
            'Retinol — Mempercepat pergantian sel kulit.',
            'Gentle Exfoliating Toner — Membantu regenerasi kulit secara berkala.'
        ],
        'avoided_ingredients': [
            'Sabun Cuci Muka dengan Scrub Kasar — Dapat melukai kulit wajah.',
            'Komedogenik Skincare — Produk dengan pelembap sangat berat yang memicu sumbatan baru.'
        ],
        'products': [
            'Cosrx AHA 7 Whitehead Power Liquid',
            'Avoskin Miraculous Refining Toner',
            'Somethinc AHA BHA PHA Peeling Solution',
            'Skintific 2% Salicylic Acid Anti Acne Serum'
        ]
    },
    'papules': {
        'name': 'Jerawat Papula',
        'emoji': get_emoji('papules'),
        'description': 'Terdeteksi adanya benjolan merah meradang pada kulit wajah Anda, namun tanpa ujung nanah (Papula). Kondisi ini disebabkan oleh sumbatan pori-pori yang terinfeksi bakteri C. acnes hingga meradang.',
        'recommended_ingredients': [
            'Benzoyl Peroxide — Sangat efektif membunuh bakteri penyebab jerawat.',
            'Sulfur — Mengurangi minyak berlebih dan membantu mengeringkan papula.',
            'Tea Tree Oil — Meredakan kemerahan dan bengkak akibat peradangan.',
            'Salicylic Acid (BHA) — Membantu mengeksfoliasi dan mengurangi sumbatan.'
        ],
        'avoided_ingredients': [
            'Physical Scrub — Gesekan fisik dapat memecah jerawat dan memperluas infeksi.',
            'Memencet Jerawat — Dapat memicu kemerahan parah, infeksi sekunder, dan bopeng.'
        ],
        'products': [
            'Skintific Acne Spot Treatment Gel',
            'Acnes Sealing Jell',
            'Somethinc Acne Shot AC-Spot Gel',
            'Sulfur Spot Lotion'
        ]
    },
    'pustules': {
        'name': 'Jerawat Pustula',
        'emoji': get_emoji('pustules'),
        'description': 'Terdeteksi adanya benjolan merah meradang yang memiliki titik putih/kuning (nanah) di puncaknya (Pustula). Kondisi ini merupakan jerawat inflamasi di mana sistem kekebalan tubuh sedang melawan infeksi bakteri di dalam pori.',
        'recommended_ingredients': [
            'Benzoyl Peroxide — Menargetkan bakteri jerawat di dalam saluran pori.',
            'Sulfur — Mengeringkan nanah pada jerawat pustula dengan cepat.',
            'Tea Tree Oil — Meredakan inflamasi dan bengkak secara alami.',
            'Salicylic Acid (BHA) — Membersihkan minyak dan sel kulit mati penyumbat.'
        ],
        'avoided_ingredients': [
            'Memecahkan Jerawat Secara Paksa — Sangat berisiko merusak jaringan kulit dan meninggalkan bekas hitam (PIH) atau bopeng.',
            'Melewatkan Pelembap — Dapat membuat kulit dehidrasi dan memicu produksi minyak lebih banyak.'
        ],
        'products': [
            'Benzolac 2.5% Gel',
            'Cosrx Acne Pimple Master Patch',
            'Somethinc Acne Shot AC-Spot Gel',
            'Acnes Sealing Jell'
        ]
    },
    'cysts': {
        'name': 'Jerawat Nodul / Kista (Jerawat Batu)',
        'emoji': get_emoji('cysts'),
        'description': 'Terdeteksi adanya benjolan besar yang terasa dalam di bawah kulit, bengkak, dan meradang hebat tanpa mata nanah yang jelas. Ini adalah jerawat batu (Nodul/Kista) yang tergolong parah karena infeksi merembet jauh ke lapisan dermis.',
        'recommended_ingredients': [
            'Mugwort — Menenangkan kulit sensitif dan meredakan peradangan hebat.',
            'Centella Asiatica — Membantu proses penyembuhan luka dan memperkuat barrier kulit.',
            'Bahan Menenangkan (Soothing) — Membantu meredakan nyeri dan kemerahan akibat bengkak.',
            'WAJIB KONSULTASI KE DOKTER SPESIALIS KULIT (Sp.KK) — Karena skincare biasa tidak bisa menembus kista.'
        ],
        'avoided_ingredients': [
            'Eksfoliasi Fisik & Kimia Berlebih — Scrub atau AHA konsentrasi tinggi dapat merusak kulit.',
            'Memencet Paksa Jerawat Batu — Sangat menyakitkan dan dipastikan akan merusak jaringan kulit dalam (menyebabkan bopeng permanen).'
        ],
        'products': [
            'Skintific Mugwort Acne Clay Mask',
            'Cosrx Centella Blemish Cream',
            'Somethinc Mugwort Tripeptide Clay Mask',
            'Sariayu Intensif Acne Care (Sulfur/Centella)'
        ]
    }
}

def count_acne_spots_native(face_crop):
    if face_crop is None or face_crop.size == 0:
        return 0
    b, g, r_chan = cv2.split(face_crop)
    redness = cv2.subtract(r_chan, g)
    blurred_red = cv2.GaussianBlur(redness, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred_red, 28, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    acne_count = 0
    face_area = face_crop.shape[0] * face_crop.shape[1]
    min_area = max(2, int(face_area * 0.00003))
    max_area = int(face_area * 0.0015)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            acne_count += 1
    return acne_count


# ── Confidence thresholds ─────────────────────────────────────────────────────
CONF_HIGH  = 0.60
CONF_LOW   = 0.30

# ── Helper for dataset statistics ─────────────────────────────────────────────
def get_dataset_stats():
    acne_dir = os.path.join(BASE_DIR, 'dataset', 'acne_types', 'AcneDataset')
    skin_dir = os.path.join(BASE_DIR, 'dataset', 'skin_types', 'Oily-Dry-Skin-Types')
    
    def count_dir(parent_path):
        if not os.path.exists(parent_path):
            return {'splits': {}, 'totals': {}, 'grand_total': 0}
        splits = ['train', 'valid', 'test']
        data = {s: {} for s in splits}
        totals = {}
        grand_total = 0
        
        for split in splits:
            split_path = os.path.join(parent_path, split)
            if not os.path.exists(split_path):
                continue
            for d in os.listdir(split_path):
                cls_path = os.path.join(split_path, d)
                if os.path.isdir(cls_path):
                    num_files = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
                    display_name = d.capitalize()
                    data[split][display_name] = num_files
                    totals[display_name] = totals.get(display_name, 0) + num_files
                    grand_total += num_files
        return {'splits': data, 'totals': totals, 'grand_total': grand_total}

    return count_dir(acne_dir), count_dir(skin_dir)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/about_ai')
def about_ai():
    return render_template('about_ai.html')

@app.route('/dataset')
def dataset():
    acne_stats, skin_stats = get_dataset_stats()
    return render_template('dataset.html', acne_stats=acne_stats, skin_stats=skin_stats)



# ── Main processing route ─────────────────────────────────────────────────────
@app.route('/process', methods=['POST'])
def process():
    load_cnn_models()

    if model_skin is None or model_acne is None:
        return render_template(
            'analyze.html',
            error_message='Model kecerdasan buatan (AI) belum siap. Pastikan "skin_type_model.tflite" dan "acne_type_model.tflite" berada di dalam direktori models/.'
        )

    if 'file' not in request.files:
        return render_template('analyze.html', error_message='Silakan unggah foto wajah.')
        
    f = request.files['file']
    if not f or f.filename == '':
        return render_template('analyze.html', error_message='Tidak ada berkas gambar yang dipilih.')

    # Read image from memory instead of saving to disk (Vercel is read-only)
    filestr = f.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return render_template('analyze.html', error_message='Format berkas tidak valid atau berkas rusak.')

    # Encode image to base64 for display in result.html
    _, buffer = cv2.imencode('.jpg', img)
    b64_string = base64.b64encode(buffer).decode('utf-8')
    image_url_b64 = f"data:image/jpeg;base64,{b64_string}"

    # ── OpenCV Haar Cascade face detection ────────────────────────────────────
    cascade_path  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade_alt   = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))

    if len(faces) == 0:
        alt_cascade = cv2.CascadeClassifier(cascade_alt)
        faces = alt_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return render_template(
            'analyze.html',
            error_message='Wajah tidak terdeteksi. Pastikan foto menunjukkan wajah manusia dengan jelas, tidak terhalang, tidak terbalik, dan bercahaya cukup.'
        )

    # Pick the largest bounding box
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    face_detected = True
    face_crop = img[y:y+h, x:x+w]

    # ── Preprocess for MobileNetV2 ──────────────────────────────────────────────
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((160, 160))
    img_arr  = np.array(face_pil, dtype='float32')
    inp      = np.expand_dims(img_arr, axis=0)
    
    # Preprocess manually since Lambda layer is removed to prevent Python version mismatch crash
    # MobileNetV2 uses (x / 127.5) - 1.0 preprocessing
    inp = (inp / 127.5) - 1.0

    # ── CNN inferences ────────────────────────────────────────────────────────
    # 1. Predict Skin Type (Dry, Normal, Oily)
    pred_skin = predict_tflite(model_skin, inp)

    # 2. Predict Acne Type (Blackheads, Cysts, None, Papules, Pustules, Whiteheads)
    pred_acne = predict_tflite(model_acne, inp)

    # ── OpenCV-based Hybrid Classification with Smooth Fusion ─────────────────
    # Convert face crop to HSV and extract channels
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    b_chan, g_chan, r_chan = cv2.split(face_crop)
    redness = cv2.subtract(r_chan, g_chan)

    mean_red = np.mean(redness)
    mean_v = np.mean(v_chan)

    # Calculate specular shine (oiliness)
    shine_mask = (v_chan > 190) & (s_chan < 80)
    shine_ratio = np.sum(shine_mask) / v_chan.size

    # Calculate texture roughness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = np.var(laplacian)

    # Count native red acne spots
    acne_spots = count_acne_spots_native(face_crop)

    # A. Hybrid Skin Type Classification with Smooth Fusion
    # Model skin output shapes: 0=dry, 1=normal, 2=oily
    p_dry = float(pred_skin[0])
    p_normal = float(pred_skin[1])
    p_oily = float(pred_skin[2])
    p_sensitive = 0.0

    # 1. Redness Fusion (Sensitive)
    # If redness is higher than 20.0, the chance of sensitive skin increases smoothly
    if mean_red > 20.0:
        sensitive_score = min(0.90, (mean_red - 20.0) / 18.0)
        p_sensitive = sensitive_score
        # Reduce other classes proportionally
        scale = 1.0 - p_sensitive
        p_dry *= scale
        p_normal *= scale
        p_oily *= scale

    # 2. Shine Fusion (Oily)
    # High shine ratio shifts probability towards oily
    if shine_ratio > 0.06:
        shine_boost = min(0.85, (shine_ratio - 0.06) / 0.10)
        p_oily = p_oily * (1.0 - shine_boost) + shine_boost
        p_normal *= (1.0 - shine_boost)
        p_dry *= (1.0 - shine_boost)
        p_sensitive *= (1.0 - shine_boost)
    elif shine_ratio < 0.03:
        # Low shine reduces oily and boosts dry
        dry_boost = min(0.80, (0.03 - shine_ratio) / 0.03)
        p_dry = p_dry * (1.0 - dry_boost) + dry_boost
        p_oily *= (1.0 - dry_boost)

    # 3. Dullness Fusion (Dry)
    # Low texture roughness (lap_var) and low shine indicates dry skin
    if lap_var < 80.0:
        flat_boost = min(0.75, (80.0 - lap_var) / 60.0)
        p_dry = p_dry * (1.0 - flat_boost) + flat_boost
        p_normal *= (1.0 - flat_boost)
        p_oily *= (1.0 - flat_boost)
        p_sensitive *= (1.0 - flat_boost)

    # Normalize Skin Probabilities
    total_p = p_dry + p_normal + p_oily + p_sensitive
    if total_p > 0:
        p_dry /= total_p
        p_normal /= total_p
        p_oily /= total_p
        p_sensitive /= total_p
    else:
        p_normal = 1.0

    skin_probs = [p_dry, p_normal, p_oily, p_sensitive]
    skin_idx = int(np.argmax(skin_probs))
    skin_class = SKIN_TYPE_CLASSES[skin_idx]
    skin_conf = float(skin_probs[skin_idx])

    # B. Hybrid Acne Type Classification with Smooth Fusion
    # Model acne output shapes: ['blackheads', 'cysts', 'none', 'papules', 'pustules', 'whiteheads']
    p_blackheads = float(pred_acne[0])
    p_cysts = float(pred_acne[1])
    p_none = float(pred_acne[2])
    p_papules = float(pred_acne[3])
    p_pustules = float(pred_acne[4])
    p_whiteheads = float(pred_acne[5])

    # If spots detected is 0, or spots count is low and none probability is high, classify as 'none'
    if acne_spots == 0 or (acne_spots <= 3 and p_none > 0.60):
        acne_class = 'none'
        acne_conf = max(0.85, p_none)
    else:
        # Acne detected, find type
        if acne_spots <= 3:
            # Blackheads vs Whiteheads
            thresh_inv = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours_black, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            black_count = sum(1 for cnt in contours_black if 1 <= cv2.contourArea(cnt) <= 15)

            thresh_w = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours_white, _ = cv2.findContours(thresh_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            white_count = sum(1 for cnt in contours_white if 1 <= cv2.contourArea(cnt) <= 15)

            if p_blackheads > 0.35 or black_count > 80:
                acne_class = 'blackheads'
                acne_conf = max(0.70, p_blackheads)
            elif p_whiteheads > 0.35 or white_count > 80:
                acne_class = 'whiteheads'
                acne_conf = max(0.70, p_whiteheads)
            else:
                if p_papules > p_pustules:
                    acne_class = 'papules'
                    acne_conf = max(0.68, p_papules)
                else:
                    acne_class = 'pustules'
                    acne_conf = max(0.68, p_pustules)
        else:
            # Significant red spots (>= 4): cysts, pustules, papules
            blurred_red = cv2.GaussianBlur(redness, (5, 5), 0)
            _, thresh_red = cv2.threshold(blurred_red, 28, 255, cv2.THRESH_BINARY)
            contours_red, _ = cv2.findContours(thresh_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            red_spot_areas = []
            face_area = face_crop.shape[0] * face_crop.shape[1]
            min_area = max(2, int(face_area * 0.00003))
            max_area = int(face_area * 0.0015)

            for cnt in contours_red:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    red_spot_areas.append(area)

            avg_area = np.mean(red_spot_areas) if red_spot_areas else 0

            # Large spots or many spots suggests cysts
            if avg_area > 120 or acne_spots >= 12 or p_cysts > 0.40:
                acne_class = 'cysts'
                acne_conf = max(0.72, p_cysts)
            else:
                # Pustules vs Papules based on spot brightness
                if mean_v > 145 or p_pustules > p_papules:
                    acne_class = 'pustules'
                    acne_conf = max(0.70, p_pustules)
                else:
                    acne_class = 'papules'
                    acne_conf = max(0.70, p_papules)

    # ── Severity logic (based on acne spots count) ────────────────────────────
    severity_name = None
    if acne_class != 'none':
        if acne_spots <= 3:
            severity_name = "Ringan (Mild)"
        elif acne_spots <= 8:
            severity_name = "Sedang (Moderate)"
        else:
            severity_name = "Parah (Severe)"

    # ── Dynamically combine recommendations ────────────────────────────────────
    rec_skin = SKIN_RECOMMENDATIONS[skin_class]
    rec_acne = ACNE_RECOMMENDATIONS[acne_class]
    
    if acne_class != 'none':
        # Single Diagnosis: Kulit Berjerawat (Jenis Jerawat)
        acne_clean_names = {
            'blackheads': 'Komedo Hitam',
            'whiteheads': 'Komedo Putih',
            'papules': 'Papula',
            'pustules': 'Pustula',
            'cysts': 'Nodul / Kista'
        }
        clean_name = acne_clean_names.get(acne_class, rec_acne['name'])
        combined_skin_type = f"Kulit Berjerawat ({clean_name})"
        combined_emoji = get_emoji(acne_class)
        primary_conf = acne_conf
        
        # Natural professional description
        severity_desc = f" dengan tingkat keparahan {severity_name.lower()}" if severity_name else ""
        combined_description = (
            f"Berdasarkan analisis kondisi wajah Anda, terdeteksi adanya **{combined_skin_type}**{severity_desc}.\n\n"
            f"{rec_acne['description']}\n\n"
            "Untuk merawat kondisi kulit ini, fokus utama adalah membersihkan pori-pori tersumbat, mengontrol produksi sebum berlebih, "
            "dan meredakan inflamasi aktif. Silakan gunakan kandungan aktif yang direkomendasikan secara rutin dan hindari "
            "bahan-bahan yang berisiko memperparah iritasi atau menyumbat pori-pori Anda."
        )
    else:
        # Single Diagnosis: Tipe Kulit Dasar
        combined_skin_type = rec_skin['name']
        combined_emoji = get_emoji(skin_class)
        primary_conf = skin_conf
        
        # Natural professional description
        combined_description = (
            f"Berdasarkan analisis kondisi wajah Anda, jenis kulit Anda tergolong sebagai **{combined_skin_type}**.\n\n"
            f"{rec_skin['description']}\n\n"
            "Disarankan untuk merawat kulit Anda dengan fokus menjaga hidrasi harian, memperkuat skin barrier, dan melindunginya "
            "dari paparan sinar matahari menggunakan sunscreen secara konsisten. Pilihlah produk yang sesuai dengan "
            "kandungan yang direkomendasikan di bawah ini."
        )

    # Combined ingredients
    combined_ingredients = list(rec_skin['recommended_ingredients'])
    if acne_class != 'none':
        for ing in rec_acne['recommended_ingredients']:
            if ing not in combined_ingredients:
                combined_ingredients.append(ing)
            
    # Combined avoided ingredients
    combined_avoided = list(rec_skin['avoided_ingredients'])
    if acne_class != 'none':
        for av in rec_acne['avoided_ingredients']:
            if av not in combined_avoided:
                combined_avoided.append(av)

    # Combined products list
    combined_products = []
    if acne_class == 'none':
        combined_products = rec_skin['products']
    else:
        # Show first 2 general skin products and all specific acne products
        combined_products = [f"[Untuk {rec_skin['name']}] {p}" for p in rec_skin['products'][:2]]
        combined_products += [f"[Untuk {rec_acne['name']}] {p}" for p in rec_acne['products']]

    # ── Confidence Gating (on primary_conf) ───────────────────────────────────
    is_very_low = primary_conf < CONF_LOW
    is_high_confidence = primary_conf >= CONF_HIGH
    image_url  = image_url_b64
    is_kista   = (acne_class == 'cysts')

    return render_template(
        'result.html',
        image_url             = image_url,
        skin_type             = combined_skin_type,
        acne_type             = rec_acne['name'],
        acne_severity         = severity_name,
        emoji                 = combined_emoji,
        confidence            = round(primary_conf * 100, 1),
        confidence_acne       = round(acne_conf * 100, 1),
        description           = combined_description,
        recommended_ingredients = combined_ingredients,
        avoided_ingredients   = combined_avoided,
        products              = combined_products,
        is_high_confidence    = is_high_confidence,
        is_very_low           = is_very_low,
        face_detected         = face_detected,
        is_kista              = is_kista,
    )

if __name__ == '__main__':
    load_cnn_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
