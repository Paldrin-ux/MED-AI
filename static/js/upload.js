// MedAI – upload.js
// Handles: drag-drop, file preview, AJAX form submission with progress bar, camera capture

(function () {
  const dropZone    = document.getElementById('dropZone');
  const fileInput   = document.getElementById('fileInput');
  const form        = document.getElementById('uploadForm');
  const submitBtn   = document.getElementById('submitBtn');
  const previewSec  = document.getElementById('previewSection');
  const previewImg  = document.getElementById('previewImg');
  const previewMeta = document.getElementById('previewMeta');
  const progressWrap = document.getElementById('progressBar');
  const progressFill = document.getElementById('progressFill');
  const progressLbl  = document.getElementById('progressLabel');
  const cameraBtn   = document.getElementById('cameraBtn');
  const cameraVideo = document.getElementById('cameraVideo');
  const captureBtn  = document.getElementById('captureBtn');
  const cameraCanvas= document.getElementById('cameraCanvas');

  // ── Drag & Drop ───────────────────────────────────────────────────────────
  ['dragenter','dragover'].forEach(ev => {
    dropZone.addEventListener(ev, e => {
      e.preventDefault(); dropZone.classList.add('drag-over');
    });
  });
  ['dragleave','drop'].forEach(ev => {
    dropZone.addEventListener(ev, e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
    });
  });
  dropZone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // ── File Input Change ─────────────────────────────────────────────────────
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    const allowed = ['image/png','image/jpeg','image/jpg'];
    const isImage = allowed.includes(file.type);

    // Preview (images only)
    if (isImage) {
      const reader = new FileReader();
      reader.onload = e => {
        previewImg.src = e.target.result;
        previewSec.classList.remove('hidden');
      };
      reader.readAsDataURL(file);
    } else {
      previewImg.src = '';
      previewSec.classList.add('hidden');
    }

    // Meta
    previewMeta.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB  ·  ${file.type || 'binary'}`;
    submitBtn.disabled = false;

    // Sync with hidden input if dropped
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
  }

  // ── AJAX Upload with Progress ─────────────────────────────────────────────
  form.addEventListener('submit', function (e) {
    e.preventDefault();

    if (!fileInput.files[0]) {
      alert('Please select a file first.');
      return;
    }

    const formData = new FormData(form);
    const xhr = new XMLHttpRequest();

    // Show progress bar
    progressWrap.classList.remove('hidden');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Uploading…';

    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        progressFill.style.width = pct + '%';
        progressLbl.textContent  = pct < 100 ? `Uploading… ${pct}%` : 'Analysing…';
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.responseURL) {
        progressLbl.textContent = 'Complete! Redirecting…';
        progressFill.style.width = '100%';
        window.location.href = xhr.responseURL;
      } else {
        progressLbl.textContent = 'Done!';
        window.location.reload();
      }
    });

    xhr.addEventListener('error', () => {
      progressLbl.textContent = 'Upload failed. Please try again.';
      submitBtn.disabled = false;
      submitBtn.textContent = '🔬 Analyse Scan';
    });

    xhr.open('POST', form.action);
    xhr.send(formData);
  });

  // ── Camera Capture ────────────────────────────────────────────────────────
  let stream = null;

  cameraBtn.addEventListener('click', async () => {
    if (stream) {
      // Stop camera
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      cameraVideo.classList.add('hidden');
      captureBtn.classList.add('hidden');
      cameraBtn.textContent = '📷 Capture from Camera';
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      cameraVideo.srcObject = stream;
      cameraVideo.classList.remove('hidden');
      captureBtn.classList.remove('hidden');
      cameraBtn.textContent = '⛔ Stop Camera';
    } catch (err) {
      alert('Camera access denied or unavailable.');
    }
  });

  captureBtn.addEventListener('click', () => {
    const ctx = cameraCanvas.getContext('2d');
    cameraCanvas.width  = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    ctx.drawImage(cameraVideo, 0, 0);

    cameraCanvas.toBlob(blob => {
      const file = new File([blob], `camera_capture_${Date.now()}.png`, { type: 'image/png' });
      handleFile(file);
      // Stop camera after capture
      stream.getTracks().forEach(t => t.stop()); stream = null;
      cameraVideo.classList.add('hidden');
      captureBtn.classList.add('hidden');
      cameraBtn.textContent = '📷 Capture from Camera';
    }, 'image/png');
  });

})();
