/**
 * Client-side search functionality using lunr.js
 */

(function () {
    'use strict';

    let searchIndex = null;
    let searchDocuments = null;
    let searchModal = null;
    let searchInput = null;
    let searchResults = null;

    /**
     * Load the pre-built search index and documents
     */
    async function loadSearchData() {
        if (searchIndex && searchDocuments) {
            return; // Already loaded
        }

        try {
            const [indexResponse, docsResponse] = await Promise.all([
                fetch('/static/js/search-index.json'),
                fetch('/static/js/search-documents.json')
            ]);

            const indexData = await indexResponse.json();
            const docsData = await docsResponse.json();

            searchIndex = lunr.Index.load(indexData);
            searchDocuments = {};
            docsData.forEach(doc => {
                searchDocuments[doc.id] = doc;
            });

            console.log('Search index loaded:', Object.keys(searchDocuments).length, 'documents');
        } catch (error) {
            console.error('Failed to load search index:', error);
        }
    }

    /**
     * Perform a search and return results
     */
    function performSearch(query) {
        if (!searchIndex || !searchDocuments || !query.trim()) {
            return [];
        }

        const trimmedQuery = query.trim();
        let allResults = [];

        // Try multiple search strategies and combine results
        const strategies = [
            // 1. Exact query (works best with stemmed words)
            trimmedQuery,
            // 2. Wildcard query (works for partial matches)
            trimmedQuery.split(/\s+/).map(term => {
                if (!/[+\-~^*:]/.test(term)) {
                    return term + '*';
                }
                return term;
            }).join(' '),
            // 3. Fuzzy search (handles typos)
            trimmedQuery.split(/\s+/).map(term => {
                if (!/[+\-~^*:]/.test(term)) {
                    return term + '~1';
                }
                return term;
            }).join(' ')
        ];

        const seenRefs = new Set();

        for (const searchQuery of strategies) {
            try {
                const results = searchIndex.search(searchQuery);
                for (const result of results) {
                    if (!seenRefs.has(result.ref)) {
                        seenRefs.add(result.ref);
                        const doc = searchDocuments[result.ref];
                        if (doc) {
                            allResults.push({
                                ...doc,
                                score: result.score
                            });
                        }
                    }
                }
            } catch (error) {
                // Strategy failed, continue to next
                console.debug('Search strategy failed:', searchQuery, error.message);
            }
        }

        // Sort by score descending and return top 10
        allResults.sort((a, b) => b.score - a.score);
        return allResults.slice(0, 10);
    }

    /**
     * Format page type for display
     */
    function formatType(type) {
        const typeLabels = {
            'blog': 'Blog',
            'talks': 'Talk',
            'teaching': 'Teaching',
            'projects': 'Project',
            'open-source': 'Open Source',
            'books': 'Book',
            'research': 'Research',
            'bio': 'Bio',
            'home': 'Home',
            'user-manual': 'User Manual'
        };
        return typeLabels[type] || type.charAt(0).toUpperCase() + type.slice(1);
    }

    /**
     * Render search results
     */
    function renderResults(results) {
        if (!searchResults) return;

        if (results.length === 0) {
            searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
            return;
        }

        const html = results.map(result => {
            const tags = result.tags && result.tags.length > 0
                ? `<div class="search-result-tags">${result.tags.slice(0, 3).map(t => `<span class="search-tag">${t}</span>`).join('')}</div>`
                : '';

            const date = result.pub_date
                ? `<span class="search-result-date">${result.pub_date}</span>`
                : '';

            const typeLabel = result.type
                ? `<span class="search-result-type">${formatType(result.type)}</span>`
                : '';

            return `
                <a href="${result.url}" class="search-result-item">
                    <div class="search-result-title">${escapeHtml(result.title)}</div>
                    <div class="search-result-meta">
                        ${typeLabel}
                        ${date}
                        ${tags}
                    </div>
                    <div class="search-result-summary">${escapeHtml(truncate(result.summary, 150))}</div>
                </a>
            `;
        }).join('');

        searchResults.innerHTML = html;
    }

    /**
     * Escape HTML to prevent XSS
     */
    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Truncate text to a maximum length
     */
    function truncate(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength).trim() + '...';
    }

    /**
     * Create the search modal
     */
    function createSearchModal() {
        const modal = document.createElement('div');
        modal.id = 'search-modal';
        modal.className = 'search-modal';
        modal.innerHTML = `
            <div class="search-modal-content">
                <div class="search-modal-header">
                    <input type="text" id="search-input" placeholder="Search the site..." autocomplete="off" />
                    <button class="search-close" aria-label="Close search">&times;</button>
                </div>
                <div id="search-results" class="search-results"></div>
                <div class="search-modal-footer">
                    <span class="search-hint">Press <kbd>ESC</kbd> to close, <kbd>Enter</kbd> to select first result</span>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        return modal;
    }

    /**
     * Open the search modal
     */
    function openSearch() {
        if (!searchModal) {
            searchModal = createSearchModal();
            searchInput = document.getElementById('search-input');
            searchResults = document.getElementById('search-results');

            // Set up event listeners
            searchInput.addEventListener('input', debounce(handleSearchInput, 200));

            searchModal.querySelector('.search-close').addEventListener('click', closeSearch);

            searchModal.addEventListener('click', (e) => {
                if (e.target === searchModal) {
                    closeSearch();
                }
            });

            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    closeSearch();
                } else if (e.key === 'Enter') {
                    const firstResult = searchResults.querySelector('.search-result-item');
                    if (firstResult) {
                        window.location.href = firstResult.href;
                    }
                }
            });
        }

        searchModal.classList.add('active');
        searchInput.value = '';
        searchResults.innerHTML = '<div class="search-hint-initial">Start typing to search...</div>';
        searchInput.focus();

        // Load search data if not already loaded
        loadSearchData();
    }

    /**
     * Close the search modal
     */
    function closeSearch() {
        if (searchModal) {
            searchModal.classList.remove('active');
        }
    }

    /**
     * Handle search input
     */
    function handleSearchInput(e) {
        const query = e.target.value;
        if (!query.trim()) {
            searchResults.innerHTML = '<div class="search-hint-initial">Start typing to search...</div>';
            return;
        }

        const results = performSearch(query);
        renderResults(results);
    }

    /**
     * Debounce function
     */
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Initialize search functionality
     */
    function init() {
        // Add click handler to search button
        const searchButton = document.getElementById('search-button');
        if (searchButton) {
            searchButton.addEventListener('click', (e) => {
                e.preventDefault();
                openSearch();
            });
        }

        // Add keyboard shortcut (Cmd/Ctrl + K)
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                openSearch();
            }
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Expose functions globally for debugging
    window.siteSearch = {
        open: openSearch,
        close: closeSearch,
        loadData: loadSearchData
    };
})();
