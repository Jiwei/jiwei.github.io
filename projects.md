---
layout: page
title: Projects
permalink: /projects/
---

<div class="project-grid">
  {%- for project in site.data.projects -%}
  <div class="project-card">
    <div class="project-thumb">
      <img src="{{ project.image | relative_url }}" alt="{{ project.title | escape }}" loading="lazy" />
    </div>
    <div class="project-body">
      <h3 class="project-title">{{ project.title | escape }}</h3>
      <p class="project-desc">{{ project.description | escape }}</p>
      <div class="project-actions">
        <a class="btn btn-github" href="{{ project.github }}" target="_blank" rel="noopener">GitHub</a>
        {%- if project.brochure -%}
        <a class="btn btn-brochure" href="{{ project.brochure | relative_url }}">Brochure</a>
        {%- endif -%}
      </div>
    </div>
  </div>
  {%- endfor -%}
</div>
