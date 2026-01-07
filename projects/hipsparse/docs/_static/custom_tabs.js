// Custom JavaScript to prevent sphinx-tabs from collapsing when clicking the active tab

document.addEventListener('DOMContentLoaded', function() {
    // Wait for sphinx-tabs to load
    setTimeout(function() {
        // Find all tab labels
        var tabLabels = document.querySelectorAll('.sphinx-tabs-tab');
        
        tabLabels.forEach(function(label) {
            // Store the original click handler
            var originalOnClick = label.onclick;
            
            // Override the click handler
            label.addEventListener('click', function(e) {
                // Check if this tab is already active
                if (this.getAttribute('aria-selected') === 'true') {
                    // Prevent the default behavior (which would collapse the tab)
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }
            }, true); // Use capture phase to intercept before sphinx-tabs
        });
    }, 100);
});

