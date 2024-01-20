function closePopup() {
    document.getElementById('custom-popup').style.display = 'none';
}

// Show the custom popup when the page loads
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('custom-popup').style.display = 'flex';
});
