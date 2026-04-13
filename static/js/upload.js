// MedAI – upload.js
// Handles: drag-drop, file preview, AJAX form submission with progress bar, camera capture

(function () {
  const dropZone     = document.getElementById('dropZone');
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

    previewMeta.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB  ·  ${file.type || 'binary'}`;
    submitBtn.disabled = false;

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

    // FIXED REDIRECT LOGIC
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 400) {
        // If the server redirected us to a results page
        if (xhr.responseURL && xhr.responseURL !== window.location.href) {
          progressLbl.textContent = 'Complete! Redirecting…';
          progressFill.style.width = '100%';
          window.location.href = xhr.responseURL;
        } else {
          // Fallback if no redirect URL is detected but status is OK
          progressLbl.textContent = 'Scan processed. Check your history.';
          setTimeout(() => { window.location.href = '/dashboard'; }, 2000);
        }
      } else {
        progressLbl.textContent = 'Error: ' + xhr.status;
        submitBtn.disabled = false;
        submitBtn.textContent = '🔬 Analyse Scan';
        console.error('Server error during scan:', xhr.responseText);
      }
    });

    xhr.addEventListener('error', () => {
      progressLbl.textContent = 'Upload failed. Please check connection.';
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
      stream.getTracks().forEach(t => t.stop()); 
      stream = null;
      cameraVideo.classList.add('hidden');
      captureBtn.classList.add('hidden');
      cameraBtn.textContent = '📷 Capture from Camera';
    }, 'image/png');
  });

})();
