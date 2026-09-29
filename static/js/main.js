/**
 * AI-Powered Resume Ranker client-side scripts.
 * Provides file selection feedback, loading state, sample JD loader,
 * interactive skill chip editing, and row expansion.
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

    // 3. Form Submit Loading State (Handles uploadForm and confirmForm)
    const forms = [document.getElementById("uploadForm"), document.getElementById("confirmForm")];
    forms.forEach((form) => {
        if (!form) return;
        form.addEventListener("submit", () => {
            const submitBtn = form.querySelector("#submitBtn");
            if (submitBtn) {
                const textContent = submitBtn.querySelector(".btn-text-content");
                const spinner = submitBtn.querySelector(".btn-spinner");
                if (textContent && spinner) {
                    textContent.style.display = "none";
                    spinner.style.display = "inline";
                    submitBtn.disabled = true;
                    submitBtn.style.opacity = "0.75";
                }
            }
        });
    });

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

    // 5. Interactive Skill Chip Management (Recruiter Workflow)
    const reqContainer = document.getElementById("requiredSkillsContainer");
    const prefContainer = document.getElementById("preferredSkillsContainer");
    const newSkillInput = document.getElementById("newSkillInput");
    const addRequiredBtn = document.getElementById("addRequiredBtn");
    const addPreferredBtn = document.getElementById("addPreferredBtn");

    function createChipElement(skillName, isRequired) {
        const chip = document.createElement("div");
        chip.className = `skill-chip ${isRequired ? "chip-required" : "chip-preferred"}`;
        chip.setAttribute("data-skill", skillName);

        const nameSpan = document.createElement("span");
        nameSpan.className = "chip-name";
        nameSpan.textContent = skillName;
        chip.appendChild(nameSpan);

        const hiddenInput = document.createElement("input");
        hiddenInput.type = "hidden";
        hiddenInput.name = isRequired ? "required_skills" : "preferred_skills";
        hiddenInput.value = skillName;
        chip.appendChild(hiddenInput);

        const toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "btn-chip-action toggle-priority";
        toggleBtn.title = isRequired ? "Move to Preferred" : "Move to Required";
        toggleBtn.textContent = "⇄";
        chip.appendChild(toggleBtn);

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "btn-chip-action remove-chip";
        removeBtn.title = "Remove requirement";
        removeBtn.textContent = "✕";
        chip.appendChild(removeBtn);

        return chip;
    }

    // Event delegation for toggle and remove
    document.addEventListener("click", (e) => {
        const toggleBtn = e.target.closest(".toggle-priority");
        if (toggleBtn) {
            const chip = toggleBtn.closest(".skill-chip");
            if (!chip) return;

            const isCurrentlyRequired = chip.classList.contains("chip-required");
            const skillName = chip.getAttribute("data-skill") || chip.querySelector(".chip-name").textContent.trim();

            if (isCurrentlyRequired && prefContainer) {
                chip.remove();
                const newChip = createChipElement(skillName, false);
                const emptyNotice = prefContainer.querySelector(".empty-chip-notice");
                if (emptyNotice) emptyNotice.remove();
                prefContainer.appendChild(newChip);
            } else if (!isCurrentlyRequired && reqContainer) {
                chip.remove();
                const newChip = createChipElement(skillName, true);
                const emptyNotice = reqContainer.querySelector(".empty-chip-notice");
                if (emptyNotice) emptyNotice.remove();
                reqContainer.appendChild(newChip);
            }
            return;
        }

        const removeBtn = e.target.closest(".remove-chip");
        if (removeBtn) {
            const chip = removeBtn.closest(".skill-chip");
            if (chip) chip.remove();
            return;
        }
    });

    // Add new skill handler
    function handleAddNewSkill(isRequired) {
        if (!newSkillInput) return;
        const skillName = newSkillInput.value.trim();
        if (!skillName) return;

        const targetContainer = isRequired ? reqContainer : prefContainer;
        if (!targetContainer) return;

        const emptyNotice = targetContainer.querySelector(".empty-chip-notice");
        if (emptyNotice) emptyNotice.remove();

        const chip = createChipElement(skillName, isRequired);
        targetContainer.appendChild(chip);
        newSkillInput.value = "";
        newSkillInput.focus();
    }

    if (addRequiredBtn) {
        addRequiredBtn.addEventListener("click", () => handleAddNewSkill(true));
    }

    if (addPreferredBtn) {
        addPreferredBtn.addEventListener("click", () => handleAddNewSkill(false));
    }

    if (newSkillInput) {
        newSkillInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleAddNewSkill(true);
            }
        });
    }
});
