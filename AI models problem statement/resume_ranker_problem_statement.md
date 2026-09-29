# AI-Powered Resume Ranker: Problem Statement

## 1. Problem Statement

Recruitment teams routinely receive far more applications than they can read carefully. A single job posting can attract hundreds of resumes, most of them unstructured documents in inconsistent layouts, lengths, and vocabularies. Recruiters usually have to make a first-pass decision on each one in a very short time, deciding who deserves a closer look and who does not. Under time pressure, this stage is inconsistent by nature. The same resume may be judged differently by different reviewers, or by the same reviewer at different points in a long screening session. Strong candidates whose resumes are unconventionally written, or simply buried deep in a large pile, can be overlooked.

Automating this stage is harder than it first appears, because a resume is not a structured record. Skills, qualifications, projects, and experience are spread across free-form text, tables, columns, and PDF layouts that do not extract cleanly. A job description is written in its own style, often mixing mandatory requirements, preferred qualifications, and general company language. The recruiter's real question is not whether the two documents share words, but whether the candidate's demonstrated background fits what the role needs.

Exact keyword matching answers only a narrow version of that question. It treats "built REST APIs with Flask" and "backend web services in Python" as unrelated, and it rewards resumes that repeat the right terms without showing any depth. It ignores context: a skill listed once in a skills section is treated the same as one applied across several projects. It also fails on abbreviations, spelling variants, and differences in phrasing ("ML" versus "machine learning", "NLP" versus "natural language processing"). Candidates who write to match a posting can score well, while those who describe equivalent experience differently score poorly. Keyword filtering therefore produces both false rejections and false confidence.

This is where Natural Language Processing and Machine Learning become relevant. Resume screening is at its core a text understanding problem: unstructured documents must be converted into text, cleaned, and normalized. Meaningful signals such as skills, qualifications, and experience must be identified, and both the candidate profile and the job requirements must be represented in a form that can be compared quantitatively. Beyond a raw score, recruiters need to know why a candidate ranked where they did, since a number that cannot be examined is hard to trust and hard to challenge.

The problem is therefore to build a decision-support system that helps recruiters prioritize applications more consistently and transparently. It would rank candidates by their relevance to a specific job description and expose the evidence behind each ranking. It is not meant to replace human judgment. Final evaluation, interviews, and hiring decisions must remain with people, and the system's value lies in helping them spend their limited attention where it is most useful.

## 2. Core Problem

Recruiters must screen large volumes of unstructured resumes against a job description, and manual review is slow and inconsistent, while simple keyword matching misses context, meaning, and equivalent phrasing. There is a need for a system that can extract and interpret resume content, measure how well each profile aligns with a role's requirements, and produce a ranked, explainable shortlist that supports, rather than replaces, human recruiters.

## 3. Key Challenges

1. **Unreliable text extraction from PDFs.** Resumes vary widely in layout, including multi-column designs, tables, icons, headers, and embedded graphics. Extraction can scramble reading order, merge sections, or drop content, so the quality of everything downstream depends on how well this step is handled.
2. **Noisy, unstructured text.** Resumes lack a standard format and contain inconsistent headings, abbreviations, formatting artifacts, and mixed content. The text must be cleaned and normalized without losing meaningful information.
3. **Identifying relevant information.** Skills, education, and experience have to be recognized within free text, including skills that are implied by project descriptions rather than listed explicitly, and distinguished from irrelevant mentions.
4. **Semantic gap between resumes and job descriptions.** The two document types use different vocabularies and writing styles. Equivalent skills expressed in different words, abbreviations, or spelling variants must not be treated as unrelated, and the comparison must weigh what the job requires most.
5. **Defining a meaningful relevance score.** Similarity between two documents does not automatically equal suitability. The scoring approach must reflect the importance of required versus optional skills and should not be inflated by keyword repetition alone.
6. **Interpretability of results.** Recruiters need to see which skills matched, which requirements are missing, and what contributed to a candidate's score. An opaque ranking is difficult to trust or verify.
7. **Fairness and responsible use.** Scoring based on text can inadvertently favor particular writing styles, formats, or backgrounds that do not reflect actual competence. The system must be framed and evaluated as an aid to human review, with its limitations made visible rather than hidden behind a confident number.

## 4. Proposed Objective

The objective is to develop an AI-assisted resume analysis and prioritization system that reads candidate resumes in PDF form, extracts and structures their relevant content, and compares each candidate profile against a given job description using NLP-based text representation and similarity measurement. The system aims to assign each candidate a relevance score, rank candidates accordingly, and explain the ranking through the matched and missing requirements. Delivered through a web interface, it is intended to reduce the effort of initial screening and make the shortlisting process more consistent and transparent, while leaving final judgment to human recruiters.

## 5. Expected Outcome

The completed system should allow a recruiter to provide a job description and a set of resumes and receive, in return, a ranked list of candidates ordered by relevance to that role. For each candidate, it should present the matching score, the skills, qualifications, and experience identified from the resume, the requirements that were matched, and the requirements that appear to be missing. Results should be presented clearly in an HR-oriented view or report so that a recruiter can quickly understand the reasoning behind each ranking, compare candidates side by side, and decide who to review first. The output is a prioritization aid: it narrows where attention goes, but the decision to shortlist, interview, or hire remains with the recruiter.
