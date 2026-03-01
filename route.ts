import { NextRequest, NextResponse } from 'next/server'
import { JSDOM } from 'jsdom'
import { spawn } from 'child_process'
import path from 'path'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

interface Article {
  id: number
  title: string
  url: string
  source: string
  content: string
  publishedAt: string
  sentiment_category?: string
  sentiment_score?: number
  sentiment_label?: string
}

// Function to run Python sentiment analysis
async function analyzeSentimentWithTransformers(articles: Article[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(process.cwd(), 'lib', 'sentiment_analyzer.py')
    
    // Prepare articles data for Python script
    const articlesData = articles.map(article => ({
      id: article.id,
      title: article.title,
      url: article.url,
      source: article.source,
      content: article.content || article.title, // Use title if content is empty
      publishedAt: article.publishedAt
    }))

    const pythonProcess = spawn('python3', [pythonScript, JSON.stringify(articlesData)], {
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let outputData = ''
    let errorData = ''

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString()
    })

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(outputData)
          console.log('Python sentiment analysis completed successfully')
          console.log('Sentiment analysis logs:', errorData) // Log Python output for debugging
          resolve(result)
        } catch (parseError) {
          console.error('Error parsing Python output:', parseError)
          console.error('Raw output:', outputData)
          reject(new Error('Failed to parse Python sentiment analysis results'))
        }
      } else {
        console.error('Python script failed with code:', code)
        console.error('Error output:', errorData)
        reject(new Error(`Python sentiment analysis failed with code ${code}`))
      }
    })

    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error)
      reject(new Error('Failed to start Python sentiment analysis'))
    })
  })
}

// Function to fetch actual publication date from article page
async function fetchArticlePublicationDate(url: string): Promise<string> {
  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      },
      signal: AbortSignal.timeout(10000) // 10 second timeout
    })
    
    if (!response.ok) {
      return new Date().toISOString() // Fallback to current time
    }

    const html = await response.text()
    const dom = new JSDOM(html)
    const document = dom.window.document

    // Try multiple common selectors for publication date
    const dateSelectors = [
      'time[datetime]',
      '.post-date time',
      '.article-date time', 
      '.publish-date',
      '[data-publish-time]',
      '.date-published',
      'meta[property="article:published_time"]',
      'meta[name="publication-date"]',
      '.post-meta time',
      '.entry-date'
    ]

    for (const selector of dateSelectors) {
      const element = document.querySelector(selector)
      if (element) {
        const dateTime = element.getAttribute('datetime') || 
                       element.getAttribute('content') || 
                       element.textContent?.trim()
        
        if (dateTime) {
          try {
            const parsedDate = new Date(dateTime)
            if (!isNaN(parsedDate.getTime()) && parsedDate.getFullYear() > 2020) {
              return parsedDate.toISOString()
            }
          } catch (e) {
            continue
          }
        }
      }
    }

    // Fallback to current time if no date found
    return new Date().toISOString()
  } catch (error) {
    console.error('Error fetching article date:', error)
    return new Date().toISOString()
  }
}

// Fetch articles from CoinTelegraph
async function fetchCoinTelegraphArticles(daysBack: number = 3): Promise<Article[]> {
  try {
    console.log('Fetching CoinTelegraph articles...')
    const response = await fetch('https://cointelegraph.com/', {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    })
    
    if (!response.ok) {
      console.error('CoinTelegraph fetch failed:', response.status)
      return []
    }

    const html = await response.text()
    const dom = new JSDOM(html)
    const document = dom.window.document
    const articles: Article[] = []
    let articleId = 1

    const postCards = document.querySelectorAll('.post-card')
    
    for (let i = 0; i < Math.min(postCards.length, 10); i++) {
      const element = postCards[i]
      
      const titleElement = element.querySelector('.post-card__title a') || element.querySelector('header a')
      if (!titleElement) continue
      
      const title = titleElement.textContent?.trim() || ''
      const relativeUrl = titleElement.getAttribute('href') || ''
      
      if (title && relativeUrl && title.length > 20) {
        const fullUrl = relativeUrl.startsWith('http') 
          ? relativeUrl 
          : `https://cointelegraph.com${relativeUrl}`

        // Fetch actual publication date from the article page
        const publishedAt = await fetchArticlePublicationDate(fullUrl)
        
        // Filter by date
        const articleDate = new Date(publishedAt)
        const cutoffDate = new Date()
        cutoffDate.setDate(cutoffDate.getDate() - daysBack)
        
        if (articleDate >= cutoffDate) {
          articles.push({
            id: articleId++,
            title,
            url: fullUrl,
            source: 'cointelegraph.com',
            content: title,
            publishedAt
          })
        }
      }
    }

    console.log(`Fetched ${articles.length} articles from CoinTelegraph`)
    return articles
  } catch (error) {
    console.error('Error fetching CoinTelegraph articles:', error)
    return []
  }
}

// Fetch articles from CryptoNews (alternative since NewsaBTC might block)
async function fetchCryptoNewsArticles(daysBack: number = 3): Promise<Article[]> {
  try {
    console.log('Fetching CryptoNews articles...')
    const response = await fetch('https://cryptonews.com/', {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    })
    
    if (!response.ok) {
      console.error('CryptoNews fetch failed:', response.status)
      return []
    }

    const html = await response.text()
    const dom = new JSDOM(html)
    const document = dom.window.document
    const articles: Article[] = []
    let articleId = 1000 // Start with different ID range

    const articleElements = document.querySelectorAll('article, .article-item, .news-item')
    
    for (let i = 0; i < Math.min(articleElements.length, 8); i++) {
      const element = articleElements[i]
      
      const titleElement = element.querySelector('h1 a, h2 a, h3 a, .title a')
      if (!titleElement) continue
      
      const title = titleElement.textContent?.trim() || ''
      const relativeUrl = titleElement.getAttribute('href') || ''
      
      if (title && relativeUrl && title.length > 20) {
        const fullUrl = relativeUrl.startsWith('http') 
          ? relativeUrl 
          : `https://cryptonews.com${relativeUrl}`

        // Default timestamp
        let publishedAt = new Date().toISOString()
        
        // Try to extract timestamp
        const timeElement = element.querySelector('.time, .date, [datetime]')
        const timeText = timeElement?.textContent?.trim() || timeElement?.getAttribute('datetime') || ''
        
        if (timeText) {
          try {
            const parsedDate = new Date(timeText)
            if (!isNaN(parsedDate.getTime())) {
              publishedAt = parsedDate.toISOString()
            }
          } catch (e) {
            // Try relative time parsing
            if (timeText.includes('hour')) {
              const hours = parseInt(timeText.match(/(\d+)/)?.[1] || '1')
              publishedAt = new Date(Date.now() - (hours * 60 * 60 * 1000)).toISOString()
            }
          }
        }

        // Filter by date
        const articleDate = new Date(publishedAt)
        const cutoffDate = new Date()
        cutoffDate.setDate(cutoffDate.getDate() - daysBack)
        
        if (articleDate >= cutoffDate) {
          articles.push({
            id: articleId++,
            title,
            url: fullUrl,
            source: 'cryptonews.com',
            content: title,
            publishedAt
          })
        }
      }
    }

    console.log(`Fetched ${articles.length} articles from CryptoNews`)
    return articles
  } catch (error) {
    console.error('Error fetching CryptoNews articles:', error)
    return []
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const daysBack = parseInt(searchParams.get('days') || '3')
    
    console.log(`Fetching real articles from the last ${daysBack} days...`)

    // Fetch articles from multiple sources
    const [coinTelegraphArticles, cryptoNewsArticles] = await Promise.all([
      fetchCoinTelegraphArticles(daysBack),
      fetchCryptoNewsArticles(daysBack)
    ])

    const allArticles = [...coinTelegraphArticles, ...cryptoNewsArticles]
    
    if (allArticles.length === 0) {
      return NextResponse.json({
        success: true,
        data: {
          positive: [],
          neutral: [],
          negative: [],
          total: 0,
          summary: {
            positive_count: 0,
            neutral_count: 0,
            negative_count: 0
          },
          message: 'No articles found for the specified date range'
        },
        timestamp: new Date().toISOString()
      })
    }

    // Sort by publication date (newest first)
    allArticles.sort((a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime())

    console.log(`Fetched ${allArticles.length} articles, running sentiment analysis...`)

    // Run sentiment analysis using Python transformers
    try {
      const sentimentResults = await analyzeSentimentWithTransformers(allArticles)
      
      console.log(`Successfully analyzed ${sentimentResults.total} articles`)
      console.log(`Sentiment distribution - Positive: ${sentimentResults.summary.positive_count}, Neutral: ${sentimentResults.summary.neutral_count}, Negative: ${sentimentResults.summary.negative_count}`)

      return NextResponse.json({
        success: true,
        data: sentimentResults,
        timestamp: new Date().toISOString()
      })
    } catch (sentimentError) {
      console.error('Sentiment analysis failed, falling back to basic analysis:', sentimentError)
      
      // Fallback: return articles without sentiment analysis
      return NextResponse.json({
        success: true,
        data: {
          positive: [],
          neutral: allArticles,
          negative: [],
          total: allArticles.length,
          summary: {
            positive_count: 0,
            neutral_count: allArticles.length,
            negative_count: 0
          },
          message: 'Sentiment analysis unavailable, showing all articles as neutral'
        },
        timestamp: new Date().toISOString()
      })
    }

  } catch (error) {
    console.error('Error in news analysis API:', error)
    
    // Fallback to empty data instead of mock data
    return NextResponse.json({
      success: false,
      data: {
        positive: [],
        neutral: [],
        negative: [],
        total: 0,
        summary: {
          positive_count: 0,
          neutral_count: 0,
          negative_count: 0
        },
        message: 'Error fetching articles'
      },
      error: 'Failed to fetch news articles',
      timestamp: new Date().toISOString()
    }, { status: 500 })
  }
}