#let col-lela = color.maroon
#let col-mehdy = color.olive
#let col-thomas = color.aqua

#let picon(c) = box(rect(fill: c, width: 8pt, height: 8pt, radius: 2pt))

#let setup(body) = {
  show heading: it => {
    v(0.8em, weak: false)
    it
  }
  body
}