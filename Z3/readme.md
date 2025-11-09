1. Opis Zaimplementowanych Modeli

Model 1:

`Model płytkiej sieni neuronowej
Architektura składa się z jednej warstwy ukrytej zawierającej 16 neuronów.
Model ten służy jako punkt odniesienia, sprawdzający efektywność prostej architektury.`

Model 2:

`Jest to głębsza, bardziej złożona sieć.
Architektura obejmuje dwie warstwy ukryte: pierwszą z 64 neuronami i drugą z 32 neuronami.
Zastosowano regularyzację Dropout w celu przeciwdziałania przeuczeniu.
`

2. Krzywe Uczenia i Dokładność

Model 1:

`Krzywe loss oraz accuracy dla zbioru treningowego i walidacyjnego są bardzo zbliżone. Obie krzywe straty zbiegają do niskich wartości.
Wniosek: Obrazuje to dobry proces uczenia. Model wykazuje dobrą generalizację i brak oznak przeuczenia.
`

Model 2:

`Wyraźna różnica między krzywymi treningowymi a walidacyjnymi. Dokładność treningowa szybko osiąga 100%, podczas gdy dokładność walidacyjna jest niższa i niestabilna.
Wniosek: Jest to symptom przeuczenia. Model zapamiętał dane treningowe, tracąc zdolność do generalizacji na nowe próbki.`

3. Wyniki i Wnioski Końcowe

`Zdecydowanie Model 1 osiągnął lepsze wyniki, uzyskując wyższą i stabilniejszą dokładność na zbiorze testowym.
Kluczowym czynnikiem jest stosunek złożoności modelu do wielkości zbioru danych.
Zbiór danych jest mały. Model 2, posiadający znacznie więcej parametrów (ze względu na większą liczbę neuronów i warstw), okazał się zbyt złożony dla tak małej ilości danych. Mimo zastosowania Dropout, doprowadził do przeuczenia.
Model 1, będąc prostszym, miał wystarczającą pojemność, aby nauczyć się wzorców w danych, a przez jego prostote że nie był w stanie zapisać szumu i specyfiki danych treningowych.`
