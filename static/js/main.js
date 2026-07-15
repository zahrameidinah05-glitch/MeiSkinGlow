document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImg = document.getElementById('preview-img');
    const btnCancelPreview = document.getElementById('btn-cancel-preview');

    const cameraSection = document.getElementById('camera-section');
    const cameraVideo = document.getElementById('camera-video');
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnCapture = document.getElementById('btn-capture');
    const btnStopCamera = document.getElementById('btn-stop-camera');

    const btnAnalyze = document.getElementById('btn-analyze');
    const analyzeForm = document.getElementById('analyze-form');
    const cameraImageInput = document.getElementById('camera-image-input');

    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');

    const resultSection = document.getElementById('result-section');
    const resultImage = document.getElementById('result-image');
    const resultSkinType = document.getElementById('result-skin-type');
    const resultConfidenceText = document.getElementById('result-confidence-text');
    const resultConfidenceCircle = document.getElementById('result-confidence-circle');
    const resultDescription = document.getElementById('result-description');
    const resultRecommended = document.getElementById('result-recommended');
    const resultAvoided = document.getElementById('result-avoided');
    const resultProducts = document.getElementById('result-products');

    let stream = null;
    let scanInterval = null;

    // --- UPLOAD HANDLERS ---
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#7835FF';
            uploadArea.style.background = 'rgba(178, 141, 255, 0.15)';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#B28DFF';
            uploadArea.style.background = 'rgba(255, 255, 255, 0.5)';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#B28DFF';
            uploadArea.style.background = 'rgba(255, 255, 255, 0.5)';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            showError('Format berkas tidak valid. Harap pilih berkas gambar.');
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            uploadPreview.style.display = 'block';
            uploadArea.style.display = 'none';
            cameraImageInput.value = ''; // Reset camera base64 input
            
            // Stop camera if running
            stopCamera();
            cameraSection.style.display = 'none';
            btnStartCamera.style.display = 'inline-block';
            
            // Enable analyze button
            btnAnalyze.disabled = false;
            hideError();
        };
        reader.readAsDataURL(file);
    }

    if (btnCancelPreview) {
        btnCancelPreview.addEventListener('click', () => {
            fileInput.value = '';
            previewImg.src = '';
            uploadPreview.style.display = 'none';
            uploadArea.style.display = 'block';
            btnAnalyze.disabled = true;
        });
    }

    // --- CAMERA HANDLERS ---
    if (btnStartCamera) {
        btnStartCamera.addEventListener('click', async () => {
            hideError();
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                cameraVideo.srcObject = stream;
                cameraSection.style.display = 'block';
                btnStartCamera.style.display = 'none';
                uploadArea.style.display = 'none';
                uploadPreview.style.display = 'none';
                fileInput.value = ''; // Clear file input
                cameraImageInput.value = '';
                btnAnalyze.disabled = true;
            } catch (err) {
                console.error("Camera access error:", err);
                showError("Gagal mengakses kamera perangkat. Pastikan Anda memberikan izin akses kamera.");
            }
        });
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            cameraVideo.srcObject = null;
        }
    }

    if (btnStopCamera) {
        btnStopCamera.addEventListener('click', () => {
            stopCamera();
            cameraSection.style.display = 'none';
            btnStartCamera.style.display = 'inline-block';
            uploadArea.style.display = 'block';
            btnAnalyze.disabled = true;
        });
    }

    if (btnCapture) {
        btnCapture.addEventListener('click', () => {
            const canvas = document.createElement('canvas');
            canvas.width = cameraVideo.videoWidth || 640;
            canvas.height = cameraVideo.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            
            // Mirror flip representation
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
            
            const dataUrl = canvas.toDataURL('image/jpeg');
            cameraImageInput.value = dataUrl;
            
            previewImg.src = dataUrl;
            uploadPreview.style.display = 'block';
            
            stopCamera();
            cameraSection.style.display = 'none';
            btnStartCamera.style.display = 'inline-block';
            btnAnalyze.disabled = false;
        });
    }

    // --- SUBMIT & ANALYSIS HANDLERS ---
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideError();
            hideResults();
            
            // Show loading animation overlay with shifting statuses
            loadingOverlay.classList.add('active');
            let step = 0;
            const statuses = [
                "Membaca data gambar...",
                "Inisialisasi modul deteksi...",
                "Memindai wajah menggunakan OpenCV Haar Cascade...",
                "Ekstraksi bagian kulit wajah...",
                "Menganalisis tekstur dan warna kulit dengan CNN...",
                "Menghitung probabilitas kecocokan..."
            ];
            
            loadingStatus.textContent = statuses[0];
            scanInterval = setInterval(() => {
                step = (step + 1) % statuses.length;
                loadingStatus.textContent = statuses[step];
            }, 800);

            const formData = new FormData(analyzeForm);

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                clearInterval(scanInterval);
                loadingOverlay.classList.remove('active');

                if (data.success) {
                    displayAnalysisResults(data);
                } else {
                    showError(data.message);
                }
            } catch (err) {
                console.error("Error during analysis fetch:", err);
                clearInterval(scanInterval);
                loadingOverlay.classList.remove('active');
                showError("Terjadi kesalahan jaringan atau server saat menganalisis foto. Harap coba lagi.");
            }
        });
    }

    function displayAnalysisResults(data) {
        resultImage.src = data.image_url;
        resultSkinType.textContent = data.skin_type;
        resultConfidenceText.textContent = `${data.confidence}%`;
        resultDescription.textContent = data.description;

        // Update SVG Progress Ring
        // Formula: stroke-dasharray = (percentage * 2 * pi * radius) / 100
        // SVG circle radius = 15.91549430918954 => Circumference = 100
        const percentage = data.confidence;
        resultConfidenceCircle.setAttribute('stroke-dasharray', `${percentage}, 100`);

        // Populate recommended ingredients
        resultRecommended.innerHTML = '';
        data.recommended_ingredients.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            resultRecommended.appendChild(li);
        });

        // Populate avoided ingredients
        resultAvoided.innerHTML = '';
        data.avoided_ingredients.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            resultAvoided.appendChild(li);
        });

        // Populate recommended products
        resultProducts.innerHTML = '';
        data.products.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'product-item';
            itemDiv.innerHTML = `
                <div class="product-icon">✨</div>
                <div><strong>${item}</strong></div>
            `;
            resultProducts.appendChild(itemDiv);
        });

        // Smooth scroll to results
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(msg) {
        if (errorAlert) {
            errorMessage.textContent = msg;
            errorAlert.style.display = 'block';
            errorAlert.scrollIntoView({ behavior: 'smooth' });
        }
    }

    function hideError() {
        if (errorAlert) {
            errorAlert.style.display = 'none';
        }
    }

    function hideResults() {
        if (resultSection) {
            resultSection.style.display = 'none';
        }
    }
});
