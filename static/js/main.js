// MedAI – main.js

// Auto-dismiss flash messages after 5s
document.querySelectorAll('.alert').forEach(el => {
  setTimeout(() => el.remove(), 5000);
});

// Animate confidence bars on load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.confidence-fill, .score-fill').forEach(el => {
    const target = el.style.width;
    el.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { el.style.width = target; });
    });
  });
});
