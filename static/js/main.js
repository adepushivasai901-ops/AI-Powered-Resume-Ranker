/**
 * AI-Powered Resume Ranker client-side scripts.
 * Provides file selection feedback, loading state, sample JD loader, and row expansion.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. File Selection Feedback
    const fileInput = document.getElementById("resumes");
    const fileSelectionInfo = document.getElementById("fileSelectionInfo");
    const fileCount = document.getElementById("fileCount");
    const fileList = document.getElementById("fileList");

    if (fileInput && fileSelectionInfo && fileCount && fileList) {
        fileInput.addEventListener("change", (e) => {
            const files = Array.from(e.target.files);
            if (files.length > 0) {
                fileCount.textContent = files.length;
                fileList.innerHTML = "";
                files.forEach((file) => {
                    const li = document.createElement("li");
                    li.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                    fileList.appendChild(li);
                });
                fileSelectionInfo.style.display = "block";
            } else {
                fileSelectionInfo.style.display = "none";
                fileList.innerHTML = "";
            }
        });
    }

    // 2. Sample Job Description Loader
    const loadSampleJdBtn = document.getElementById("loadSampleJdBtn");
    const jobDescriptionInput = document.getElementById("job_description");

    if (loadSampleJdBtn && jobDescriptionInput) {
        loadSampleJdBtn.addEventListener("click", () => {
            jobDescriptionInput.value =
                "Position: Senior AI / Machine Learning Engineer\n\n" +
                "Requirements & Responsibilities:\n" +
                "- 3+ years of professional software engineering and machine learning experience.\n" +
                "- Expert proficiency in Python and SQL.\n" +
                "- Hands-on experience with NLP, PyTorch, Scikit-learn, and Pandas.\n" +
                "- Experience deploying scalable models using Docker, AWS, and REST APIs / FastAPI.\n" +
                "- Familiarity with CI/CD pipelines, Git version control, and PostgreSQL databases.\n" +
                "- Strong analytical, problem-solving, and cross-functional communication skills.";
        });
    }

    // 3. Form Submit Loading State
    const uploadForm = document.getElementById("uploadForm");
    const submitBtn = document.getElementById("submitBtn");

    if (uploadForm && submitBtn) {
        uploadForm.addEventListener("submit", (e) => {
            const textContent = submitBtn.querySelector(".btn-text-content");
            const spinner = submitBtn.querySelector(".btn-spinner");

            if (textContent && spinner) {
                textContent.style.display = "none";
                spinner.style.display = "inline";
                submitBtn.disabled = true;
                submitBtn.style.opacity = "0.75";
            }
        });
    }

    // 4. Toggle Scoring Explanation Details
    const toggleButtons = document.querySelectorAll(".toggle-explanation");
    toggleButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");
            const targetRow = document.getElementById(targetId);
            if (targetRow) {
                const isHidden = targetRow.style.display === "none";
                targetRow.style.display = isHidden ? "table-row" : "none";
                btn.textContent = isHidden ? "Hide" : "Details";
            }
        });
    });
});
