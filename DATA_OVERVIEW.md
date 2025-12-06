# DATA OVERVIEW - Basketball in Africa QA Dataset

This document provides a complete overview of the data used in this HW6 project.

---

## 📊 Dataset Statistics

### Questions & Answers
- **Total QA pairs**: 159
- **Question types**: 
  - Factoid: 53 questions (33.3%)
  - List: 53 questions (33.3%)
  - Multiple Choice: 53 questions (33.3%)

### Corpus Documents
- **Total documents**: 53
- **Total words**: ~11,263 words
- **Average document length**: ~212 words
- **Format**: Plain text (.txt)
- **Source URLs**: Included in evidence.tsv

---

## 📝 Sample Questions by Type

### Factoid Questions (Short Answer)
Questions that expect a specific fact, name, date, or short phrase as answer.

**Examples:**
1. **Q**: Who is the current commissioner of the Basketball Africa League?
   - **A**: Amadou Gallo Fall

2. **Q**: In what year was FIBA Africa established?
   - **A**: 1961

3. **Q**: When did basketball's presence in Africa begin?
   - **A**: Mid-20th century

4. **Q**: How often is AfroBasket held?
   - **A**: Every four years

5. **Q**: What publisher is associated with "Sport and Apartheid South Africa"?
   - **A**: Routledge

### List Questions (Multiple Items)
Questions that expect multiple items or examples as answer.

**Examples:**
1. **Q**: List three teams that have won the BAL championship since its inaugural season in 2021.
   - **A**: Zamalek, US Monastir, Al Ahly (possible answers)

2. **Q**: Name three African national teams that have been dominant in AfroBasket history.
   - **A**: Angola, Tunisia, Senegal (possible answers)

3. **Q**: Name two sports introduced by European colonial authorities.
   - **A**: Cricket, Football (possible answers)

4. **Q**: List two barriers that historically restricted women's participation in basketball in Africa.
   - **A**: Cultural norms, Lack of infrastructure (possible answers)

5. **Q**: Name two emerging basketball nations mentioned as pushing for greater representation.
   - **A**: Rwanda, Mozambique (possible answers)

### Multiple Choice Questions
Questions with 4 options (A, B, C, D) where one is correct.

**Examples:**
1. **Q**: Which organization co-organizes the Basketball Africa League (BAL) alongside FIBA?
   - A) UEFA  
   - B) **NBA Africa** ✓
   - C) IOC  
   - D) FIFA

2. **Q**: Which city serves as the headquarters of FIBA Africa?
   - A) Lagos  
   - B) **Cairo** ✓
   - C) Abidjan  
   - D) Nairobi

3. **Q**: Which colonial region used cricket as a tool for elite formation?
   - A) French West Africa  
   - B) **British India** ✓
   - C) Caribbean  
   - D) East Africa

4. **Q**: Which two countries are mentioned as leaders in women's basketball achievements in Africa?
   - A) Egypt and Morocco  
   - B) **Nigeria and Senegal** ✓
   - C) Kenya and Ghana  
   - D) Tunisia and Algeria

5. **Q**: What is the primary purpose of allowing two African countries to participate in global events?
   - A) Increase revenue  
   - B) **Promote inclusivity and showcase broader talent** ✓
   - C) Reduce competition  
   - D) Follow European models

---

## 📚 Sample Corpus Documents

### Example 1: basketball_africa_league.txt
**Source**: Wikipedia  
**URL**: https://en.wikipedia.org/wiki/Basketball_Africa_League  
**Length**: ~300 words

**Content Preview:**
```
Title: Basketball Africa League (BAL)
Source: Wikipedia

INTRODUCTION
The Basketball Africa League (BAL) is the premier men's club basketball 
league in Africa, co-organized by NBA Africa and FIBA. It launched in 2019 
and held its inaugural season in 2021, featuring 12 teams each season.

HISTORICAL CONTEXT
The NBA and FIBA announced the BAL on February 16, 2019, during NBA All-Star 
weekend. The first season—delayed by COVID-19—was played in a bubble in 
Kigali in May 2021, with Zamalek winning the inaugural title over US Monastir.

FORMAT & QUALIFICATION
The BAL currently features 12 teams: seven qualify directly as national 
champions, while five earn spots via the Road to BAL qualifiers...

KEY FACTS
- Organizing bodies: NBA Africa & FIBA
- Number of teams: 12; season typically runs March–June
- Commissioner: Amadou Gallo Fall
- Current champions (2025): Al Ahli Tripoli (1st title)
- Champions by season: 2021 Zamalek; 2022 US Monastir; 2023 Al Ahly; 
  2024 Petro de Luanda; 2025 Al Ahli Tripoli
```

### Example 2: fiba_africa_history.txt
**Source**: FIBA Official Website  
**URL**: https://about.fiba.basketball/en/regions/africa/history  
**Length**: ~250 words

**Content Preview:**
```
Title: FIBA Africa - History
Source: FIBA Official Website

ESTABLISHMENT
FIBA Africa was established in 1961 to govern basketball across the African 
continent. It serves as one of FIBA's five regional confederations.

HEADQUARTERS
Cairo, Egypt serves as the headquarters of FIBA Africa...

MAJOR COMPETITIONS
- AfroBasket (Men's and Women's)
- FIBA Africa Championship
- Youth Championships (U16, U18)
```

### Example 3: colonial_history_sport.txt
**Source**: The Conversation  
**URL**: https://theconversation.com/how-colonial-history-shaped-bodies-and-sport...  
**Length**: ~280 words

**Content Preview:**
```
Title: Colonial History and Sport
Source: The Conversation

COLONIAL INFLUENCE
European colonial powers used sport as tools of governance, identity 
formation, and social control in colonized regions. Cricket, football, 
and other sports were introduced to reinforce colonial hierarchies...

BRITISH COLONIALISM
In British colonies, cricket was used as a tool for elite formation and 
to establish cultural dominance...

FRENCH COLONIALISM
French colonial authorities introduced football and athletics as part 
of their "civilizing mission"...
```

---

## 🔗 Evidence Mapping

The `evidence.tsv` file maps each question to its source document.

**Format**: `<source_url>\t<filename>`

**Sample Mappings:**

| Question # | Source URL | Document Filename |
|------------|------------|-------------------|
| 1 | https://en.wikipedia.org/wiki/Basketball_Africa_League | basketball_africa_league.txt |
| 2 | https://about.fiba.basketball/en/regions/africa/history | fiba_africa_history.txt |
| 3 | https://theconversation.com/how-colonial-history-shaped... | colonial_history_sport.txt |
| 4 | https://selamta.ethiopianairlines.com/inspiration/... | womens_basketball_africa.txt |
| 5 | https://www.africabasket.net/articles/two-countries... | africabasket_two_countries.txt |

---

## 📂 Complete Corpus Document List

The corpus contains 53 documents covering various aspects of Basketball in Africa:

### Topic Categories:

**1. Basketball Africa League (BAL) - 15 documents**
- basketball_africa_league.txt
- bal_africa_sports_economy_summary.txt
- bal_afd_education_inclusion_summary.txt
- bal_first_scouting_combine_summary.txt
- bal_gender_equality_initiatives_summary.txt
- bal_global_investment_summary.txt
- bal_linking_african_teams_nba_summary.txt
- bal_partners_playoffs_finals_summary.txt
- bal_rdb_multi_year_extension_summary.txt
- investing_in_bal_private_investment_summary.txt
- nba_academy_bal_us_college_summary.txt
- ...and more

**2. FIBA Africa & Competitions - 8 documents**
- fiba_africa_history.txt
- fiba_africa_sub_zones_summary.txt
- fiba_africa_youth_competitions_summary.txt
- fiba_africa_youth_development.txt
- afrobasket_summary.txt
- african_games_3x3_basketball_summary.txt
- espn_africa_fiba_broadcast_deal_summary.txt
- ...and more

**3. African Basketball Players - 8 documents**
- dikembe_mutombo_summary.txt
- giannis_antetokounmpo_summary.txt
- hakeem_olajuwon_summary.txt
- joel_embiid_summary.txt
- pascal_siakam_summary.txt
- serge_ibaka_summary.txt
- gab_gre_lasme_summary.txt
- luol_deng_south_sudan_summary.txt

**4. Women's Basketball - 4 documents**
- womens_basketball_africa.txt
- womens_basketball_league_africa.txt
- top_african_female_players_summary.txt
- bal_gender_equality_initiatives_summary.txt

**5. Infrastructure & Development - 8 documents**
- basketball_africa_infrastructure_summary.txt
- basketball_africa_societal_impact_summary.txt
- nba_africa_opportunity_courts_summary.txt
- nba_africa_opportunity_international_summary.txt
- giants_of_africa_camps_summary.txt
- giants_of_africa_2025_kigali_summary.txt
- youth_development_basketball_summary.txt
- bwb_east_africa_summary.txt

**6. Historical & Cultural Context - 4 documents**
- colonial_history_sport.txt
- colonialism_sport_book_summary.txt
- rise_of_african_basketball.txt
- rise_and_rise_african_basketball.txt

**7. Leagues & Organizations - 6 documents**
- basketball_national_league_summary.txt
- nigerian_premier_league_summary.txt
- youzou_elite_basketball_summary.txt
- top_10_african_basketball_clubs_summary.txt
- nba_fiba_african_dev_league.txt
- basketball_influence_africa_summary.txt

---

## 📈 Data Quality Characteristics

### Question Characteristics:
- **Specificity**: Questions require information from corpus (not general knowledge)
- **Diversity**: Cover historical, current, players, leagues, infrastructure, social impact
- **Difficulty**: Range from simple facts to complex analysis
- **Format variation**: Factoid, list, and multiple choice formats

### Answer Characteristics:
- **Concise**: Most answers are 1-10 words (factoid) or 2-5 items (list)
- **Verifiable**: All answers can be verified from source documents
- **Multiple valid answers**: List questions may have multiple correct responses
- **Format consistency**: Answers are tab-separated in TSV

### Corpus Characteristics:
- **Source diversity**: Wikipedia, official organizations, news articles, academic sources
- **Temporal coverage**: Historical to current (2025) information
- **Geographic scope**: Pan-African with focus on major basketball nations
- **Content structure**: Most documents have clear sections and key facts

---

## 🎯 Use Cases for This Dataset

This dataset is designed to test:

1. **Retrieval Quality**: Can the system find relevant documents for each question?
2. **Generator Accuracy**: Can the system extract/synthesize correct answers from documents?
3. **Format Handling**: Can the system handle different question types appropriately?
4. **Domain Knowledge**: Does the system understand Basketball Africa context?
5. **Multi-hop Reasoning**: Some questions require information from multiple documents

---

## 🔍 Sample RAG Pipeline Flow

**Example Question**: "Who is the current commissioner of the Basketball Africa League?"

### Step 1: Retrieval (Top-3 Documents)
1. `basketball_africa_league.txt` (cosine similarity: 0.89)
2. `bal_africa_sports_economy_summary.txt` (similarity: 0.72)
3. `fiba_africa_history.txt` (similarity: 0.65)

### Step 2: Context Formation
Concatenate retrieved documents:
```
The Basketball Africa League (BAL) is the premier men's club basketball 
league in Africa, co-organized by NBA Africa and FIBA...
Commissioner: Amadou Gallo Fall...
```

### Step 3: Generation
**Prompt to Generator**:
```
Context: [retrieved documents]
Question: Who is the current commissioner of the Basketball Africa League?
Answer:
```

**Generated Answer**: "Amadou Gallo Fall"

### Step 4: Output
Save to TSV:
```
Amadou Gallo Fall\tbasketball_africa_league.txt, bal_africa_sports_economy_summary.txt, fiba_africa_history.txt
```

---

## 📁 File Format Details

### question.tsv
- **Format**: `<question>\t<type>`
- **Encoding**: UTF-8 with Windows line endings (CRLF)
- **Size**: 17 KB
- **Lines**: 159

### answer.tsv
- **Format**: `<answer1>\t<answer2>\t...\t\t\t\t` (tab-separated, multiple possible answers)
- **Encoding**: UTF-8 with Windows line endings (CRLF)
- **Size**: 6.1 KB
- **Lines**: 159

### evidence.tsv
- **Format**: `<source_url>\t<filename>`
- **Encoding**: UTF-8 with Windows line endings (CRLF)
- **Size**: 20 KB
- **Lines**: 159

### corpus/*.txt
- **Format**: Plain text with structured sections
- **Encoding**: UTF-8
- **Total Size**: ~300 KB (all 53 files)
- **Average Size**: ~5.7 KB per file

---

## ✅ Data Completeness Checklist

- ✅ All 159 questions have corresponding answers
- ✅ All 159 questions have evidence mappings
- ✅ All 53 corpus documents exist and are readable
- ✅ Question types are balanced (factoid, list, multiple choice)
- ✅ Sources include diverse, authoritative references
- ✅ Document content covers all question topics
- ✅ Data is clean with proper encoding

---

## 🎓 For Report Writing

When documenting your dataset in Section 2 of your report:

**Key Points to Include:**
1. **Size & Composition**: 158 QA pairs, 53 documents, ~11K words
2. **Question Distribution**: Balanced across 3 types (factoid, list, multiple choice)
3. **Topic Coverage**: BAL, FIBA Africa, players, infrastructure, history, women's basketball
4. **Sources**: Wikipedia, FIBA, news outlets, academic sources
5. **Quality**: All questions require retrieval (not answerable from general knowledge)
6. **Verification**: Evidence mappings link questions to source documents

**Sample Table for Report:**
| Metric | Value |
|--------|-------|
| Total QA Pairs | 159 |
| Corpus Documents | 53 |
| Total Words | 11,263 |
| Avg Document Length | 212 words |
| Question Types | Factoid (53), List (53), Multiple Choice (53) |
| Topics Covered | 7 major categories |
| Source Types | Official sites, Wikipedia, news, academic |

---

This dataset provides a comprehensive foundation for evaluating RAG systems on the Basketball in Africa domain. The diversity of question types, balanced distribution, and quality source documents make it suitable for rigorous evaluation of retrieval and generation components.
