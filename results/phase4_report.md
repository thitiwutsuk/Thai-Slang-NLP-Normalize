# Phase 4: Evaluation & Error Analysis

## Accuracy / F1 comparison

| Setup | Accuracy | Macro-F1 |
|---|---|---|
| Raw text | 0.7435 | 0.6428 |
| Normalized text | 0.7432 | 0.6551 |

## Per-class F1

| label | raw | normalized |
|---|---|---|
| pos | 0.4975 | 0.5293 |
| neu | 0.7944 | 0.7900 |
| neg | 0.7895 | 0.7916 |
| q | 0.4898 | 0.5094 |

## Confusion matrices

Raw:
```
          pos   neu   neg     q
true pos  197   258    21     2
true neu   99  1238   102    14
true neg   17   138   527     1
true q      1    30     2    24
```

Normalized:
```
          pos   neu   neg     q
true pos  226   220    30     2
true neu  132  1187   118    16
true neg   17   117   545     4
true q      1    28     1    27
```

## Net effect on test-set predictions

- **174** examples flipped from wrong (raw) to correct (normalized)
  - 116 had an actual `normalize_thai()` correction (elongation/emoji/slang)
  - 58 had **no correction at all** — the only difference was that the normalized pipeline tokenizes and rejoins with spaces, which changes how WangchanBERTa's own SentencePiece tokenizer segments the input
- **175** examples flipped from correct (raw) to wrong (normalized) — the cost side of the same trade
- Net: **-1** more correct predictions on the test set

## Example cases normalization fixed

- raw: `มาชมเรวๆๆ`
  - normalized: `มา ชม เร วๆ`
  - true=`neu` raw_pred=`pos` → normalized_pred=`neu`
- raw: `เบอร์8อิโดนิเชียเล่นดีจิงๆแต่ไทยก็ไม่คมถ้าเล่นแบบนี้ก็ยุแต่ในอาเชียนเถอะคับ คลองบอลก็น้อยกว่าเขา โค๊ชใหม่ก็ไม่เข้าใจเท่ากับโค๊ชคนไทยด้วยกัน บอลไทยกำลังกลับไปยุในยุกเดิม มืดมน ขอบคุณเบียร์ช้างที่สนับสนุน ผมอย่างเป็นทางการ`
  - normalized: `เบอร์ 8 อี โด  เชีย เล่น ดี  แต่ ไทย ก็ ไม่ คม ถ้า เล่น  ก็ อยู่ แต่ ใน อา เชีย น เถอะ ครับ คลอง  ก็ น้อยกว่า เขา โค๊ช ใหม่ ก็ ไม่ เข้าใจ เท่ากับ โค๊ช คนไทย ด้วยกัน  ไทย กำลัง กลับ ไป อยู่ ใน อยู่ ก เดิม มืดมน ขอบคุณ เบียร์ ช้าง ที่ สนับสนุน ผม อย่าง เป็นทางการ`
  - true=`neu` raw_pred=`neg` → normalized_pred=`neu`
- raw: `แม่หญิงใช้ อายไลเนอร์ มิสทีน รุ่น ซุปเปอร์แบล็ค ฟิคซ์อายไลเนอร์ ข้าอยากใช้เพราะ เส้นคม ดำสนิท ติดทนนาน ให้ดวงตาสวยเฉียบคมมีเสน่ห์ และมิสทีนของออเจ้า ไม่เคยทำให้ข้าผิดหวัง`
  - normalized: `แม่ หญิง ใช้ อาย ไลเนอร์ มิสทีน รุ่น ซุปเปอร์ แบล็ค ฟิคซ์ อาย ไลเนอร์ ข้า อยาก ใช้ เพราะ  คม ดำ สนิท ติด ทน นาน ให้ ดวงตา สวย เฉียบคม มีเสน่ห์ และ มิสทีน ของ ออเจ้า ไม่ เคย ทำให้ ข้า ผิดหวัง`
  - true=`neu` raw_pred=`pos` → normalized_pred=`neu`
- raw: `จะหาครีมกันแดดเนื้อเหลว ๆ ไม่หนึบสำหรับเด็กของการ์นิเย่ที่ไทยได้ไหมนะ ชอบมาก`
  - normalized: `จะ หา ครีมกันแดด เนื้อ เหลว ๆ ไม่ หนึบ สำหรับ เด็ก ของ การ์  เย่ ที่ ไทย ได้ ไหม นะ ชอบ มาก`
  - true=`q` raw_pred=`neg` → normalized_pred=`q`
- raw: `มันไม่ใช่ไข่พยาธิ #แต่มันคือตัวอ่อนพยาธิเลย!!!!!!!! https://m.facebook.com/story.php?story_fbid=10154963647346455&id=51620386454`
  - normalized: `มัน ไม่ ใช่ ไข่ พยาธิ แต่ มัน คือ ตัวอ่อน พยาธิ เลย !`
  - true=`neg` raw_pred=`neu` → normalized_pred=`neg`
- raw: `เราก็ใช้ Mazda 3 ปีนี้เข้าปีที่4 ไม่เคยซ่อมเลย มีแต่เอาไปเช็คปีละครั้ง ซื้อที่ออสเตรเลีย`
  - normalized: `เรา ก็ ใช้ Mazda 3 ปี นี้ เข้า ปี ที่ 4 ไม่ เคย ซ่อม เลย มี แต่ เอา ไป เช็ก ปี ละ ครั้ง ซื้อ ที่ ออสเตรเลีย`
  - true=`pos` raw_pred=`neu` → normalized_pred=`pos`
- raw: `ได้หมดถ้าสดชื่น...555`
  - normalized: `ได้ หมด ถ้า สดชื่น .[pos_emoji]`
  - true=`pos` raw_pred=`neu` → normalized_pred=`pos`
- raw: `อยากลองใช้ K+ ของลาโรชมากๆเลย ไม่รู้จะช่วยสิวอุดตันได้จริงมั้ย`
  - normalized: `อยาก ลอง ใช้ K + ของ ลา โร ชมา กๆ เลย ไม่ รู้ จะ ช่วย สิว อุดตัน ได้ จริง ไหม`
  - true=`q` raw_pred=`pos` → normalized_pred=`q`

## Example cases normalization broke

- raw: `เจ้ว่าการ์นิเย่แอบแรงนิสหน่อย เคยใช้โยเกิร์ตไม๊ พรุ้งนี้ลองพอกหน้าดูทำให้หน้าสบายขึ้น นุ่มขึ้น หายไวไวน๊าาาา`
  - normalized: `เจ๊ ว่า การ์  เย่ แอ บแรง  สห น่อย เคย ใช้ โยเกิร์ต ไม๊ พ รุ้ง นี้ ลอง พอก หน้า ดู ทำ ให้หน้า สบาย ขึ้น นุ่ม ขึ้น หาย ไว ไว นะ`
  - true=`neg` raw_pred=`neg` → normalized_pred=`neu`
- raw: `สำหรับงาน Transmisson : The Spirit of Warrior ต้องขวดนี้ค่ะ Jack Daniel's Tennessee Whiskey honey ป็นวิสกี้ที่โดดเด่นในเรื่องของความหอมหวานราวกับน้ำผึ้ง(เดือน5) เพราะนำเอา Honey Liqueur ถึง 4 ชนิดผสมเข้าไป ทำให้ได้รสเข้มข้นเหมือนกับดื่มน้ำผึ้งอยู่จริง ๆ แต่ไม่ได้หวานมากจนบาดคอนะคะ ผู้ชายก็ชอบผู้หญิงก็ต้องบอกว่าใช่!! แล้วเจอกันค่ะ <3`
  - normalized: `สำหรับ งาน Transmisson : The Spirit of Warrior ต้อง ขวด นี้ ค่ะ Jack Daniel 's Tennessee Whiskey honey ป็น วิสกี้ ที่ โดดเด่น ใน เรื่อง ของ ความ หอม หวาน ราวกับ น้ำผึ้ง ( เดือน 5 ) เพราะ นำ เอา Honey Liqueur ถึง 4 ชนิด ผสม เข้าไป ทำ ให้ได้  เข้มข้น เหมือนกับ ดื่ม น้ำผึ้ง อยู่ จริง ๆ แต่ ไม่ ได้ หวาน มาก จน บาดคอ นะคะ ผู้ชาย ก็ ชอบ ผู้หญิง ก็ ต้อง บอ กว่า ใช่ !! แล้ว เจอกัน ค่ะ <3`
  - true=`pos` raw_pred=`pos` → normalized_pred=`neu`
- raw: `ก็เพิ่งรู้ว่ามีผ้าอนามัยแบบสอด😂`
  - normalized: `ก็ เพิ่ง รู้ ว่า มี ผ้าอนามัย แบบ สอด [pos_emoji]`
  - true=`neu` raw_pred=`neu` → normalized_pred=`neg`
- raw: `เอ็มเคไง มึงอยากกินไม่ใช่หรอ`
  - normalized: `เอ็ม  ไง มึง อยาก กิน ไม่ ใช่ เหรอ`
  - true=`pos` raw_pred=`pos` → normalized_pred=`neu`
- raw: `สิ่งที่ช่วยเติมเต็มรสชาติความอร่อยในมื้ออาหารของเราคงจะเป็น "เบียร์ช้าง" เย็นๆๆๆ ไม่ว่าจะหนาวและจะร้อน เพียงแค่ได้ดื่มเบียร์ช้างกับเพื่อนที่รู้ใจไม่ว่าความทุกข์จะหนักหนาแค่ไหนก็ไม่ได้เป็นอุปสรรค ยิ่งได้กินอาหารที่มีรสชาติอร่อยฟังเพลงเพราะๆกับศิลปินที่ชอบก็ทำให้ฟินไปอีก`
  - normalized: `สิ่ง ที่ ช่วย เติมเต็ม รสชาติ ความ อร่อย ใน มื้อ อาหาร ของ เรา คงจะ เป็น " เบียร์ ช้าง " เย็น ๆ ไม่ ว่า จะ หนาว และ จะ ร้อน เพียงแค่ ได้ ดื่ม เบียร์ ช้าง กับ เพื่อน ที่ รู้ใจ ไม่ ว่า ความทุกข์ จะ หนักหนา แค่ ไหน ก็ ไม่ ได้ เป็น  ยิ่ง ได้ กิน อาหาร ที่ มี รสชาติ อร่อย ฟังเพลง เพราะ ๆ กับ ศิลปิน ที่ ชอบ ก็ ทำให้ ฟิน ไป อีก`
  - true=`pos` raw_pred=`pos` → normalized_pred=`neu`

## Limitations

- **Slang dictionary coverage is incomplete.** It only contains corrections mined from MultiLexNorm++'s annotated corpus (17k+ entries, ≥3 occurrences). Slang not present there passes through unchanged — e.g. `ชิมิ` ("ใช่ไหม") isn't in the dictionary, so PyThaiNLP still mis-tokenizes it into `['ชิ', 'มิ']`.
- **A meaningful share of the improvement is a tokenization-spacing artifact, not a linguistic correction** — 58/174 of the fixed cases had zero logged corrections. This means part of the raw-vs-normalized gap reflects PyThaiNLP's word segmentation helping WangchanBERTa's subword tokenizer, not the dictionary/elongation logic specifically — worth separating out in any follow-up ablation.
- **Sarcasm and other pragmatic phenomena are out of scope.** normalize_thai() only fixes surface form (spelling, elongation, emoji); it can't recover meaning that depends on tone or context, e.g. sarcastic "ดีมากเลยค่ะ" said about bad service.
- **Emoji mapping is coarse.** All emoji collapse into 3 tags (`[pos_emoji]`/`[neg_emoji]`/`[emoji]`) rather than preserving per-emotion nuance (e.g. angry vs. sad both become `[neg_emoji]`).
- **Metrics are on a single run**, not averaged over multiple seeds — the reported deltas (macro-F1 +1.2pp) are directionally meaningful given the shared setup, but exact magnitudes would benefit from repeated runs to establish variance.