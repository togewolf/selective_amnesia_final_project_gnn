#import "config.typ": *

#import "@preview/charged-ieee:0.1.3": ieee

#set heading(numbering: "1.1.1")

#set par(justify: true)

#show ref: it => {
  if it.element != none and it.element.func() == figure {
    it
  } else {
    show regex("\d+"): set text(blue)
    it
  }
}

// --- TITLE PAGE ---
#block(height: 100%, width: 100%)[
  #set align(center)
  #v(3cm)
  #text(size: 23pt, weight: "bold")[Selective Amnesia: Targeted Unlearning in different Conditional Generative Models]
  
  #v(1cm)
  #text(size: 16pt)[Generative Neural Networks for the Sciences]
  #v(-0.3cm)
  #text(size: 14pt)[Prof. Köthe — WS 2025/26]
  
  #v(1cm)
  #text(size: 18pt, weight: "semibold")[Final Project Report]
  

  #v(1cm)
  #text(size: 12pt)[Date: 25.03.2026]

  #v(8cm)

  #text(size: 12pt, weight: "bold")[GitHub Repository\ ]
  #text(fill: blue, style: "italic")[#link("https://github.com/togewolf/selective_amnesia_final_project_gnn")]
  #v(1cm)
  #text(size: 12pt)[Written by:]
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [*Lela Eigenrauch*\ #text(size: 9pt)[Matr. Nr: 4161820]],
    [*Mehdy Shinwari*\ #text(size: 9pt)[Matr. Nr: 4152444]],
    [*Thomas Wolf*\ #text(size: 9pt)[Matr. Nr: 4273318]]
  )   
]

#pagebreak()
// --- TABLE OF CONTENTS ---
#show outline.entry.where(level: 1): it => { v(12pt, weak: true); strong(it) }
#outline(depth: 2, indent: auto)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#show: ieee.with(
  title: [Project Report for Generative Neural Networks],
  authors: (
    (name: "Lela Eigenrauch", department: picon(col-lela),email: "lela.eigenrauch@stud.uni-heidelberg.de"),
    (name: "Mehdy Shinwari", department: picon(col-mehdy),email: "mehdy.shinwari01@stud.uni-heidelberg.de"),
    (name: "Thomas Wolf", department: picon(col-thomas),email: "thomas.wolf01@stud.uni-heidelberg.de"),
  ),
)

// --- MAIN CONTENT ---
#include "sections/abstract.typ"
#include "sections/introduction.typ"
#include "sections/background.typ"
#include "sections/methods.typ"
#include "sections/experiments_results.typ"
#include "sections/conclusion.typ"

#pagebreak()
#heading(outlined: true, numbering: none)[References #picon(col-lela)] 
#bibliography("refs.bib", style: "ieee", title: none)
#set page(columns: 1)
#include "sections/appendix.typ"