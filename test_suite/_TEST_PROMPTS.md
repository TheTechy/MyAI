
# PROMPT SETS

### SET 1 - General
- I am planning a 5 day trip to Miami soon and would like to know some key highlights I should visit while I am there? Create an itinerary and display it here please.
---
- I am planning a three day trip to the lake District soon and would like to know some key highlights I should visit while I am there?
---
- What's the atomic structure for Cobalt?
---
- What are the three longest rivers in the world?
---
- If you had to explain quantum entanglement to a 14 year old, how would you do it?
---
- Give me a detailed explanation of why the sky is blue?

### SET 2 - Code
- Write a JavaScript debounce utility that supports both leading and trailing edge execution, includes a cancel method, and save it as debounce.js
---
- Write a Python class that implements a thread-safe LRU cache with a configurable max size. Include get, put and clear methods with full docstrings and save it as lru_cache.py
---
- Explain the difference between async/await and Task.Run() in C#. When would you use one over the other?
---
- Write a SQL script that creates a normalised database schema for a library management system. Include tables for books, authors, members and loans with appropriate foreign keys and indexes. Save it as library_schema.sql
---
- Create a C# console application that reads a CSV file, calculates the average, min and max of a numeric column and outputs the results. Save it as csv_analyser.cs

### SET 3 - File generation
- Please create a Word document (docx), no more than 4 pages, that describes the Battle of Britain. Focus on the tactics used by the British to defeat the much larger German force.
---
- Write a professional Word document (docx) outlining a C# coding standards guide for a development team. Cover naming conventions, error handling, and commenting best practices.
---
- Create an Excel spreadsheet (xlsx) comparing the planets of the Solar System. Include columns for planet name, diameter, distance from the Sun, number of moons, and whether it has rings.
---
- Create a projected monthly budget for a small coffee shop in its first year of operation. Export this data as a downloadable `.xlsx` file with clean column layouts and basic sum formulas.
---
- Generate an XLSX spreadsheet for a fictional retail company. Requirements:
  * Exactly 36 rows of data
  * Columns: OrderID, Date, CustomerSegment, Region, ProductCategory, Quantity, UnitPrice, Discount, TotalRevenue
  * Ensure realistic distributions and valid calculations.
---
- Draft a standard Mutual Non-Disclosure Agreement (NDA) into a Word file. Use placeholders like [Company Name] and [Effective Date]. Ensure the tone is strictly legal and authoritative. Ensure it ends with clean signature blocks. Organize it into numbered sections:
  * Definition of Confidential Information
  * Obligations of the Receiving Party
  * Term and Termination.

### SET 4 - File understanding
- I have uploaded a Word document. Please read the content thoroughly and provide the following response inline, do not generate a file.
  * The Executive Summary: A concise, 3-4 sentence overview of the document's main purpose and conclusion.
  * Key Pillars: A bulleted list of the top 3-5 core themes or arguments presented, including a brief 1-sentence explanation for each.
  * Action Items / Takeaways: Any explicit or implied next steps, recommendations, or major conclusions found in the text.
  * Target Audience Note: Who is the intended audience for this document, and what is the overall tone (e.g., formal, technical, persuasive)?
---
- I have uploaded an Excel spreadsheet (or CSV data). Please read the content thoroughly and provide the following feedback.
  * Data Schema & Structure: Describe what this dataset represents. Identify the key columns/attributes and the apparent layout.
  * Top 3 High-Level Trends: Based on the numbers or categories, what are the 3 most obvious trends, correlations you can detect?
  * Anomalies or Outliers: Are there any glaring spikes, drops, missing values, or formatting inconsistencies that look unusual or worth investigating?

### SET 5 - Weather skill
 * Will it rain tomorrow?
---
 * What's the weather like in Manchester right now?
---
 * What's the weather forecast for London this week?
---
 * Should I bring an umbrella to Carlisle UK today?
---
 * Is it t-shirt and shorts weather in London right now?

### SET 6 - Calculator skill
 * What is £85 increased by 12.5%?
---
 * Convert 100 miles to km
---
 * What is 35% off £899?
---
 * How many lbs is 75kg?
---
 * Convert 72°F to Celsius
---
 * What is (450 + 550) * 12?

### SET 7 - Date & Time
 * What time is it in Tokyo?
---
 * What time is it in New York?
---
 * What day was 6th June 1944?
---
 * How many days until Christmas?
---
 * How old is someone born on 15th March 1990?
---
 * What day was it on 19th Jan 1873?

### SET 8 - Currency
 * Convert £500 to USD
---
 * What is the GBP to EUR exchange rate?
---
 * How much is €250 in pounds?
---
 * What was the GBP to USD rate on 1st January 2020?
---
 * What is £100 in euros?

### SET 9 - Web Search
 * Who won this years F1 Canadian Grand Prix?
---
 * Give me today’s UK news headlines.
---
 * What is the current price of Bitcoin in GBP?
---
 * What are the current Premier League standings?
---
 * What were the top 3 AI stories this week?

### SET 10 - Image Skill
 * Resize this image to 800×600
---
 * Crop this image to a 400×400 square from the centre
---
 * Scale this image down to 50% of its original size
---
 * Rotate this image 90 degrees clockwise
---
 * Flip this image horizontally
---
 * Rotate this image 45 degreesColour & Tone
---
 * Convert this image to greyscale
---
 * Increase the brightness of this image by 50%
---
 * Boost the contrast of this image
---
 * Sharpen this image
---
 * Convert this image to JPEG format
---
 * Save this image as a PNG with maximum compression
---
 * Apply a blur effect to this image
---
 * Add a sepia tone to this image
---
 * Invert the colours of this image

### SET 11 - Directions
 * Give me directions from Liverpool to Birmingham.
---
 * I need directions from Leeds to Edinburgh Castle
---

### SET 12 - Multi-Turn
*Goal: Check if the application retains session context across sequential turns naturally.*
(Turn 1)
 * My name is Sarah, I run a boutique digital marketing agency in London, and my biggest struggle right now is client retention. Can you give me three general tips to improve retention?

(Turn 2)
 * What's is the weather looking like for the rest of the week?

(Turn 3)
 * Based on our conversation earlier, write a short, polite email I could send to a client who hasn't replied to my last three check-ins. Keep the tone aligned with my specific business type and location.