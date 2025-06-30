// Титульный лист (ГОСТ Р 7.0.11-2001, 5.1)
#let title-page(
  organization,
  organization-short,
  organization-logo,
  author-full-name1,
  course-number1,
  isu-number1,
  author-full-name2,
  course-number2,
  isu-number2,
  thesis-title,
  supervisor-full-name,
  supervisor-regalia,
  thesis-city,
  year
) = {

  // Настройки
  set page(numbering: {})
  set par(
    leading: 0.7em, justify: false
  )
  set text(hyphenate: false)
  set align(center)

  v(3em)

  // Огранизация
  if organization-logo.len() > 0 {
    image(organization-logo, width: 25%)
  } else [
    LOGO
  ]
  [Факультет Систем Управления и Робототехники]

  v(7em)

  text(16pt)[*#thesis-title*]

  v(5em)


  // text(16pt)[#author-full-name]
  grid(columns: (2fr, 1fr),
    [
        #set par(justify: true)
        #set text(hyphenate: false)
        #set align(left)

        #h(2em)
        *Аннотация* – Проект охватывает сборку платформы LEGO Mindstorms с ROS 2, разработку одометрии, тестирование ICP и AMCL, внедрение визуального SLAM (ArUco, VO SLAM) и сравнительный анализ результатов.

        *Ключевые слова:* SLAM; ROS 2; LEGO Mindstorms; Одометрия; ICP; AMCL; Visual SLAM; VO SLAM; ArUco-маркеры; rosbag; Картирование; Локализация; LiDAR; Визуальная одометрия
    ],
    []
  )

  v(3em)

  align(right)[

    *Практику проходили*: \
    студент #course-number1 курса, #isu-number1 \
    #author-full-name1 \
    студент #course-number2 курса, #isu-number2 \
    #author-full-name2

    *Руководитель*: \
    #supervisor-regalia \
    #supervisor-full-name
  ]

  v(1fr)

  [#thesis-city #year]
}