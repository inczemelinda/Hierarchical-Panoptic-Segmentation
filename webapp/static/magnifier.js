(function () {
    "use strict";

    const ZOOM = 2.5;
    const LOUPE_SIZE = 220;

    document.addEventListener("DOMContentLoaded", init);

    function init() {
        const toggle = document.getElementById("magnifier-toggle");
        const images = document.querySelectorAll(".results-grid img");
        if (!toggle || images.length === 0) {
            return;
        }

        const loupe = document.createElement("div");
        loupe.className = "loupe";
        loupe.style.width = LOUPE_SIZE + "px";
        loupe.style.height = LOUPE_SIZE + "px";
        document.body.appendChild(loupe);

        let active = false;

        function setActive(next) {
            active = next;
            toggle.classList.toggle("active", active);
            toggle.setAttribute("aria-pressed", active ? "true" : "false");
            document.body.classList.toggle("magnifier-active", active);
            if (!active) {
                loupe.style.display = "none";
            }
        }

        toggle.addEventListener("click", function () {
            setActive(!active);
        });

        function updateLoupe(event, img) {
            if (!active) {
                return;
            }
            const rect = img.getBoundingClientRect();
            const mx = event.clientX - rect.left;
            const my = event.clientY - rect.top;

            if (mx < 0 || my < 0 || mx > rect.width || my > rect.height) {
                loupe.style.display = "none";
                return;
            }

            // Account for object-fit: cover — find the rendered natural-image rect
            // inside the displayed box.
            const nW = img.naturalWidth || rect.width;
            const nH = img.naturalHeight || rect.height;
            const scale = Math.max(rect.width / nW, rect.height / nH);
            const offsetX = (nW * scale - rect.width) / 2;
            const offsetY = (nH * scale - rect.height) / 2;

            const bgW = nW * scale * ZOOM;
            const bgH = nH * scale * ZOOM;
            const r = LOUPE_SIZE / 2;
            const bgPosX = r - (mx + offsetX) * ZOOM;
            const bgPosY = r - (my + offsetY) * ZOOM;

            loupe.style.backgroundImage = "url(" + img.src + ")";
            loupe.style.backgroundSize = bgW + "px " + bgH + "px";
            loupe.style.backgroundPosition = bgPosX + "px " + bgPosY + "px";
            loupe.style.left = (event.clientX - r) + "px";
            loupe.style.top = (event.clientY - r) + "px";
            loupe.style.display = "block";
        }

        images.forEach(function (img) {
            img.addEventListener("mousemove", function (e) {
                updateLoupe(e, img);
            });
            img.addEventListener("mouseenter", function (e) {
                updateLoupe(e, img);
            });
            img.addEventListener("mouseleave", function () {
                if (active) {
                    loupe.style.display = "none";
                }
            });
        });

        // Hide the loupe if the cursor leaves the predictions grid altogether
        const grid = document.querySelector(".results-grid");
        if (grid) {
            grid.addEventListener("mouseleave", function () {
                loupe.style.display = "none";
            });
        }

        // Hide on scroll because the loupe position is viewport-fixed
        window.addEventListener("scroll", function () {
            if (active) {
                loupe.style.display = "none";
            }
        }, { passive: true });
    }
})();
