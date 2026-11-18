# Local LLM comparison

5 queries, top 5 songs each, temperature 0.

| model | valid JSON | valid ids | coverage | answers w/ invented numbers | avg s | tok/s |
|---|---|---|---|---|---|---|
| qwen2.5:3b | 5/5 | 5/5 | 100% | 0/5 | 14.3 | 20.2 |
| llama3.2:3b | 5/5 | 5/5 | 76% | 0/5 | 14.0 | 18.6 |
| qwen2.5:1.5b | 5/5 | 5/5 | 76% | 0/5 | 10.5 | 38.3 |
| llama3.2:1b | 5/5 | 5/5 | 100% | 0/5 | 9.0 | 31.5 |

## qwen2.5:3b

### mellow songs for late-night studying

json=True, ids=True, coverage=100%, invented=none, 15.4s, 20.1 tok/s

```json
{
  "summary": "All songs are categorized under the 'study' genre and share a calm, mellow, melancholic, and moody sound with a slow tempo.",
  "picks": [
    {
      "id": "S1",
      "why": "Matches the request with a calm, mellow, melancholic, moody, and slow tempo sound suitable for late-night studying."
    },
    {
      "id": "S2",
      "why": "Completely instrumental with a calm, mellow, melancholic, moody, and slow tempo, perfect for a silent study session."
    },
    {
      "id": "S3",
      "why": "Shares the 'study' genre and has a calm, mellow, melancholic, moody, and danceable sound, perfect for a relaxing study session."
    },
    {
      "id": "S4",
      "why": "Matches the request with a calm, mellow, melancholic, moody, and slow tempo sound suitable for late-night studying."
    },
    {
      "id": "S5",
      "why": "Matches the request with a calm, mellow, melancholic, moody, and slow tempo sound suitable for late-night studying."
    }
  ]
}
```

### angry heavy metal for the gym

json=True, ids=True, coverage=100%, invented=none, 15.1s, 20.2 tok/s

```json
{
  "summary": "All songs are heavy metal with high energy and upbeat tempo.",
  "picks": [
    {
      "id": "S1",
      "why": "High-energy and intense sound matches the request for angry music suitable for the gym."
    },
    {
      "id": "S2",
      "why": "Same high-energy and intense sound as S1, ensuring consistent mood for the gym environment."
    },
    {
      "id": "S3",
      "why": "Tempo is very close to the target, maintaining the desired intensity for the gym setting."
    },
    {
      "id": "S4",
      "why": "While in the death metal genre, it is not suitable for the request as it is not for dancing and has a different mood."
    },
    {
      "id": "S5",
      "why": "Close match in terms of energy, tempo, and mood, providing a strong alternative to S1 and S2."
    }
  ]
}
```

### happy upbeat latin dance party

json=True, ids=True, coverage=100%, invented=none, 11.9s, 20.4 tok/s

```json
{
  "summary": "All songs are upbeat salsa tracks with high danceability and energy.",
  "picks": [
    {
      "id": "S1",
      "why": "Matches the request with its energetic and cheerful sound, perfect for a happy upbeat dance party."
    },
    {
      "id": "S2",
      "why": "Adds to the party vibe with its upbeat tempo and cheerful sound, complementing the other songs."
    },
    {
      "id": "S3",
      "why": "Provides a slightly different flavor of salsa with its upbeat tempo, enhancing the party atmosphere."
    },
    {
      "id": "S4",
      "why": "Maintains the party mood with its energetic and cheerful sound, adding variety to the playlist."
    },
    {
      "id": "S5",
      "why": "Enhances the party with its very danceable and groovy sound, ensuring a lively dance floor."
    }
  ]
}
```

### sad acoustic breakup songs

json=True, ids=True, coverage=100%, invented=none, 16.1s, 20.1 tok/s

```json
{
  "summary": "The retrieved songs are all categorized as sad and have acoustic qualities.",
  "picks": [
    {
      "id": "S1",
      "why": "Matches the request with its sad genre and acoustic sound, having moderate energy and a danceable, moody, and upbeat tempo."
    },
    {
      "id": "S2",
      "why": "Also sad and acoustic, with moderate energy and a fast tempo, fitting the request for sad breakup songs."
    },
    {
      "id": "S3",
      "why": "Sad and acoustic, with moderate energy and a fast tempo, aligning with the listener's request for sad breakup songs."
    },
    {
      "id": "S4",
      "why": "Sad and acoustic, with moderate energy and a fast tempo, matching the listener's request for sad breakup songs."
    },
    {
      "id": "S5",
      "why": "While sad, it has acoustic qualities and a fast tempo, but also has positive and upbeat elements, which do not fit the request for sad breakup songs."
    }
  ]
}
```

### soft instrumental piano for falling asleep

json=True, ids=True, coverage=100%, invented=none, 13.1s, 20.0 tok/s

```json
{
  "summary": "All songs are soft instrumental piano pieces suitable for falling asleep.",
  "picks": [
    {
      "id": "S1",
      "why": "Matches the request with very calm, quiet, dark, and sad sounds, perfect for falling asleep."
    },
    {
      "id": "S2",
      "why": "Matches the request with very calm, quiet, dark, and sad sounds, perfect for falling asleep."
    },
    {
      "id": "S3",
      "why": "Matches the request with very calm, quiet, dark, and sad sounds, perfect for falling asleep."
    },
    {
      "id": "S4",
      "why": "Matches the request with very calm, quiet, dark, and sad sounds, perfect for falling asleep."
    },
    {
      "id": "S5",
      "why": "Matches the request with very calm, quiet, dark, and sad sounds, perfect for falling asleep."
    }
  ]
}
```

## llama3.2:3b

### mellow songs for late-night studying

json=True, ids=True, coverage=100%, invented=none, 13.9s, 18.5 tok/s

```json
{
  "summary": "These songs are mellow and suitable for late-night studying.",
  "picks": [
    {
      "id": "S1",
      "why": "It has a calm and mellow sound, and its tempo is close to the target tempo, making it suitable for studying."
    },
    {
      "id": "S2",
      "why": "It is an instrumental track with a slow tempo, fitting for a calm and focused study session."
    },
    {
      "id": "S3",
      "why": "Although it has a slightly upbeat tempo, its overall sound is mellow and melancholic, making it suitable for studying."
    },
    {
      "id": "S4",
      "why": "It has a slow tempo and a mellow sound, making it suitable for a calm and focused study session."
    },
    {
      "id": "S5",
      "why": "It has a slow tempo and a mellow sound, similar to S1, making it suitable for studying."
    }
  ]
}
```

### angry heavy metal for the gym

json=True, ids=True, coverage=80%, invented=none, 16.7s, 18.6 tok/s

```json
{"summary": "Recommended songs for 'angry heavy metal for the gym': S1, S2, S3, S5", "picks": [{"id": "S1", "why": "High-energy, intense, and upbeat tempo make 'Breaker' a great fit for the gym, while its heavy-metal genre aligns with the listener's request."}, {"id": "S2", "why": "Similar to 'Breaker', 'The Krochik' shares a high-energy and intense sound, with a close match to the target's valence and energy."}, {"id": "S3", "why": "With a tempo within 1 BPM of the target and a high match score, 'El Secreto Más Perverso' is a strong contender for the gym playlist, offering a similar heavy-metal sound."}, {"id": "S5", "why": "Like 'Breaker', 'El Viaje' has a high-energy and intense sound, with a tempo close to the target and a high match score, making it a suitable choice for the gym playlist."}]}
```

### happy upbeat latin dance party

json=True, ids=True, coverage=60%, invented=none, 12.2s, 18.8 tok/s

```json
{
  "summary": "These songs are upbeat and energetic Latin dance party tracks.",
  "picks": [
    {
      "id": "S1",
      "why": "It has a happy and cheerful sound, and its tempo is close to the target tempo of a happy upbeat Latin dance party."
    },
    {
      "id": "S2",
      "why": "It has a high match score and its tempo is very close to the target tempo of a happy upbeat Latin dance party."
    },
    {
      "id": "S5",
      "why": "It has a groovy sound and its danceability is very close to the target, making it a great fit for a happy upbeat Latin dance party."
    }
  ]
}
```

### sad acoustic breakup songs

json=True, ids=True, coverage=80%, invented=none, 13.2s, 18.7 tok/s

```json
{"summary": "Recommended songs for a sad acoustic breakup: S1, S2, S3, S4", "picks": [{"id": "S1", "why": "matches the target genre and has a moderate energy, melancholic, and moody sound, with a tempo close to the target tempo of 14 BPM"}, {"id": "S2", "why": "matches the target genre and has a fast tempo, which may be a departure from the typical slow tempo of a breakup song, but still conveys a sense of sadness and urgency"}, {"id": "S3", "why": "has a melancholic and moody sound, with a tempo close to the target tempo of 14 BPM, and conveys a sense of longing and loss"}, {"id": "S4", "why": "matches the target genre and has a fast tempo, which may be a departure from the typical slow tempo of a breakup song, but still conveys a sense of desperation and longing"}]}
```

### soft instrumental piano for falling asleep

json=True, ids=True, coverage=60%, invented=none, 13.9s, 18.6 tok/s

```json
{"summary": "Recommended soft instrumental piano songs for falling asleep: S1, S2, S4", "picks": [{"id": "S1", "why": "same genre (piano) and similar sound (very calm, quiet, dark, sad, not for dancing, acoustic, mid-tempo, instrumental, no vocals) as the target song"}, {"id": "S2", "why": "same genre (new-age) and similar sound (very calm, quiet, dark, sad, not for dancing, acoustic, mid-tempo, instrumental, no vocals) as the target song"}, {"id": "S4", "why": "similar genre (disney ~ piano) and similar sound (very calm, quiet, dark, sad, not for dancing, acoustic, mid-tempo) as the target song"}]}
```

## qwen2.5:1.5b

### mellow songs for late-night studying

json=True, ids=True, coverage=100%, invented=none, 11.8s, 38.5 tok/s

```json
{
  "summary": "The listener requested mellow songs for late-night studying. The recommended songs include 'Morning Coffee' by inbetweenjobs, which is calm, mellow, and has a slow tempo, matching the request. 'Sleepover' by Timothy Infinite is also a good fit, as it is calm, mellow, and has a slow tempo, and matches the request closely. 'Night Sky' by Faime is another option, as it is chill and has a slow tempo, which aligns with the request. '4 Course Meal' by Matt Large is another choice, as it is calm, mellow, and has a slow tempo, and matches the request closely. 'String Lights' by Dimension 09 is also a good fit, as it is calm, mellow, and has a slow tempo, and matches the request closely.",
  "picks": [
    {
      "id": "S1",
      "why": "The song is calm, mellow, and has a slow tempo, which is perfect for late-night studying."
    },
    {
      "id": "S2",
      "why": "The song is calm, mellow, and has a slow tempo, which is perfect for late-night studying."
    },
    {
      "id": "S3",
      "why": "The song is chill and has a slow tempo, which is perfect for late-night studying."
    },
    {
      "id": "S4",
      "why": "The song is calm, mellow, and has a slow tempo, which is perfect for late-night studying."
    },
    {
      "id": "S5",
      "why": "The song is calm, mellow, and has a slow tempo, which is perfect for late-night studying."
    }
  ]
}
```

### angry heavy metal for the gym

json=True, ids=True, coverage=80%, invented=none, 8.8s, 38.3 tok/s

```json
{
  "summary": "The listener requests an angry, heavy-metal song for the gym. The first song, 'Breaker' by Accept, matches the request closely with high energy, intense, and melancholic sound, and is within the target range for genre, energy, tempo, and danceability.",
  "picks": [
    {
      "id": "S1",
      "why": "The song matches the request for genre, energy, tempo, and danceability, and is within the target range for acousticness and valence."
    },
    {
      "id": "S2",
      "why": "The song matches the request for genre, energy, tempo, and danceability, and is within the target range for acousticness and valence."
    },
    {
      "id": "S3",
      "why": "The song matches the request for genre, energy, tempo, and danceability, and is within the target range for acousticness and valence."
    },
    {
      "id": "S5",
      "why": "The song matches the request for genre, energy, tempo, and danceability, and is within the target range for acousticness and valence."
    }
  ]
}
```

### happy upbeat latin dance party

json=True, ids=True, coverage=100%, invented=none, 10.9s, 38.2 tok/s

```json
{
  "summary": "The listener's request for a happy, upbeat Latin dance party is well met by the songs 'Juancito Trucupey' by Dimension Latina, 'La Guagua' by Celia Cruz, 'El Cayuco' by Tito Puente, 'Por Eso' by Tito Allen, and 'Sigue Tu Camino' by Dimension Latina.",
  "picks": [
    {
      "id": "S1",
      "why": "The song 'Juancito Trucupey' by Dimension Latina is a salsa track that matches the request for a happy, upbeat Latin dance party. It has a high danceability, energy, and tempo, making it perfect for a lively dance party."
    },
    {
      "id": "S2",
      "why": "The song 'La Guagua' by Celia Cruz is a salsa track that matches the request for a happy, upbeat Latin dance party. It has a high energy, tempo, and danceability, making it perfect for a lively dance party."
    },
    {
      "id": "S3",
      "why": "The song 'El Cayuco' by Tito Puente is a salsa track that matches the request for a happy, upbeat Latin dance party. It has a high energy, tempo, and danceability, making it perfect for a lively dance party."
    },
    {
      "id": "S4",
      "why": "The song 'Por Eso' by Tito Allen is a salsa track that matches the request for a happy, upbeat Latin dance party. It has a high energy, tempo, and danceability, making it perfect for a lively dance party."
    },
    {
      "id": "S5",
      "why": "The song 'Sigue Tu Camino' by Dimension Latina is a salsa track that matches the request for a happy, upbeat Latin dance party. It has a high energy, tempo, and danceability, making it perfect for a lively dance party."
    }
  ]
}
```

### sad acoustic breakup songs

json=True, ids=True, coverage=40%, invented=none, 6.0s, 38.4 tok/s

```json
{
  "summary": "The listener's request for sad acoustic breakup songs is met by several options, with S1 and S2 being the most fitting choices.",
  "picks": [
    {
      "id": "S1",
      "why": "The song 'run away (feat. Chelsea Collins)' is a sad acoustic breakup song with moderate energy and a melancholic sound, matching the listener's request."
    },
    {
      "id": "S2",
      "why": "The song 'Where You Went Wrong' by Kado is a sad acoustic breakup song with moderate energy, melancholic sound, and a fast tempo, which aligns well with the listener's request."
    }
  ]
}
```

### soft instrumental piano for falling asleep

json=True, ids=True, coverage=60%, invented=none, 15.1s, 37.9 tok/s

```json
{
  "summary": "The listener requested a soft instrumental piano track for falling asleep. The most fitting songs are 'Blund' by Jacob David, 'Nocturnal Mind in d Minor - Piano Septet Version' by Yiruma, and 'Transformation - Remastered 2018' by Chorus - Beauty And the Beast.",
  "picks": [
    {
      "id": "S1",
      "why": "The song 'Blund' by Jacob David is a piano-based track with a very calm and quiet sound, matching the request for a soft instrumental piano track. It has a valence score within 0.00 of the target, indicating a positive mood, and danceability within 0.04 of the target, suggesting it's not for dancing. The tempo is within 5 BPM of the target, and it's acoustic, instrumental, and has no vocals, which aligns with the request for a slow tempo and no vocals."
    },
    {
      "id": "S2",
      "why": "The song 'Nocturnal Mind in d Minor - Piano Septet Version' by Yiruma is a new-age piano track with a very calm and quiet sound, matching the request for a soft instrumental piano track. It has a danceability score within 0.02 of the target, indicating it's not for dancing, and tempo within 2 BPM of the target, suggesting a slow tempo. The energy is within 0.04 of the target, and acousticness within 0.06 of the target, which aligns with the request for a slow tempo and no vocals."
    },
    {
      "id": "S4",
      "why": "The song 'Transformation - Remastered 2018' by Chorus - Beauty And the Beast is a Disney piano track with a very calm and quiet sound, matching the request for a soft instrumental piano track. It has a valence score within 0.01 of the target, indicating a positive mood, and danceability within 0.02 of the target, suggesting it's not for dancing. The tempo is within 7 BPM of the target, and acousticness within 0.08 of the target, which aligns with the request for a slow tempo and no vocals."
    }
  ]
}
```

## llama3.2:1b

### mellow songs for late-night studying

json=True, ids=True, coverage=100%, invented=none, 8.7s, 31.6 tok/s

```json
{"summary": "Mellow songs for late-night studying", "picks": [
{"id": "S1", "why": "Calm and soothing, with a slow tempo and melancholic sound, making it perfect for studying"},
{"id": "S2", "why": "Instrumental and danceable, with a slow tempo and no vocals, ideal for creating a relaxing atmosphere"},
{"id": "S3", "why": "Upbeat and danceable, with a calm and melancholic sound, suitable for a late-night study session"},
{"id": "S4", "why": "Slow and danceable, with a calm and melancholic sound, and a similar genre to study music"},
{"id": "S5", "why": "Instrumental and slow, with a calm and soothing sound, making it perfect for studying in the evening"}]}
```

### angry heavy metal for the gym

json=True, ids=True, coverage=100%, invented=none, 11.6s, 30.6 tok/s

```json
{
  "summary": "Recommended for gym use: 'Breaker' by Accept, 'The Krochik' by Ricardo Iorio, 'El Secreto Más Perverso' by Lörihen, 'Angry Mob Justice' by The Acacia Strain, 'El Viaje' by Lörihen",
  "picks": [
    {
      "id": "S1",
      "why": "High-energy and intense, with a melancholic tone, making it perfect for a workout"
    },
    {
      "id": "S2",
      "why": "High-energy and intense, with a melancholic tone, and a tempo that matches the intensity of a gym session"
    },
    {
      "id": "S3",
      "why": "High-energy and intense, with a melancholic tone, and a tempo that's close to the target"
    },
    {
      "id": "S4",
      "why": "High-energy and intense, with a dark and sad tone, but not for dancing, and a tempo that's close to the target"
    },
    {
      "id": "S5",
      "why": "High-energy and intense, with a melancholic tone, and a tempo that matches the intensity of a gym session"
    }
  ] }
```

### happy upbeat latin dance party

json=True, ids=True, coverage=100%, invented=none, 8.2s, 31.9 tok/s

```json
{
  "summary": "The listener requested a happy upbeat Latin dance party playlist.",
  "picks": [
    {
      "id": "S1",
      "why": "This song is a classic salsa track with a high energy level and a cheerful vibe, making it perfect for a dance party."
    },
    {
      "id": "S2",
      "why": "This song is a high-energy salsa track with a fast tempo and a happy, cheerful sound, making it ideal for a dance party."
    },
    {
      "id": "S3",
      "why": "This song has a similar energy level to S2, with a fast tempo and a happy, cheerful sound, making it a great fit for a dance party."
    },
    {
      "id": "S4",
      "why": "This song has a similar tempo to S3, with a fast tempo and a happy, cheerful sound, making it a great fit for a dance party."
    },
    {
      "id": "S5",
      "why": "This song is a high-energy salsa track with a fast tempo and a danceable sound, making it perfect for a Latin dance party."
    }
  ] }
```

### sad acoustic breakup songs

json=True, ids=True, coverage=100%, invented=none, 9.4s, 31.8 tok/s

```json
{
  "summary": "sad acoustic breakup songs",
  "picks": [
    {
      "id": "S1",
      "why": "This song has a melancholic tone and a moderate energy level, making it a fitting choice for a sad breakup song."
    },
    {
      "id": "S2",
      "why": "This song has explicit lyrics and a fast tempo, which could be suitable for a more upbeat and emotional breakup song."
    },
    {
      "id": "S3",
      "why": "This song has a danceable tempo and upbeat energy, making it a good fit for a sad breakup song that wants to be more energetic."
    },
    {
      "id": "S4",
      "why": "This song has a calm and mellow tone, which could be suitable for a more introspective and emotional breakup song."
    },
    {
      "id": "S5",
      "why": "This song has a positive and upbeat tone, which could be suitable for a more optimistic and hopeful breakup song."
    }
  ] }
```

### soft instrumental piano for falling asleep

json=True, ids=True, coverage=100%, invented=none, 7.1s, 31.6 tok/s

```json
{"summary": "Soft instrumental piano for falling asleep", "picks": [{"id": "S1", "why": "Calming and soothing sound, suitable for relaxation and sleep"}, {"id": "S2", "why": "Mellow and peaceful sound, perfect for meditation and calming the mind"}, {"id": "S3", "why": "Slow and gentle tempo, ideal for unwinding and promoting relaxation"}, {"id": "S4", "why": "Similar genre to S3, with a calming and soothing sound"}, {"id": "S5", "why": "Slow and calming tempo, suitable for sleep and relaxation"}]}
```
