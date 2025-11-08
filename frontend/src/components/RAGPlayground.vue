<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
    <div class="max-w-7xl mx-auto">
      <!-- Header -->
      <div class="text-center mb-8">
        <h1 class="text-4xl font-bold text-gray-900 mb-2">RAG Explorer</h1>
        <p class="text-gray-600">Explore and experiment with Retrieval-Augmented Generation</p>
      </div>

      <!-- Main tabs -->
      <div class="bg-white rounded-lg shadow-lg">
        <!-- Tab navigation -->
        <div class="border-b border-gray-200">
          <nav class="flex -mb-px">
            <button
              v-for="tab in tabs"
              :key="tab.id"
              @click="activeTab = tab.id"
              :class="[
                'px-6 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              ]"
            >
              {{ tab.name }}
            </button>
          </nav>
        </div>

        <!-- Tab content -->
        <div class="p-6">
          <!-- Search Tab -->
          <div v-if="activeTab === 'search'" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <!-- Collection selector -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Collection</label>
                <select v-model="searchConfig.collectionId" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option v-for="col in collections" :key="col.id" :value="col.id">
                    {{ col.name }}
                  </option>
                </select>
              </div>

              <!-- Embedding provider -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Embedding Provider</label>
                <select v-model="searchConfig.embeddingProviderId" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option v-for="provider in embeddingProviders" :key="provider.id" :value="provider.id">
                    {{ provider.name }}
                  </option>
                </select>
              </div>

              <!-- Search type -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Search Type</label>
                <select v-model="searchConfig.searchType" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option value="semantic">Semantic (Vector Only)</option>
                  <option value="hybrid">Hybrid (Vector + Keyword)</option>
                </select>
              </div>

              <!-- Top K -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Results (Top K)</label>
                <input
                  v-model.number="searchConfig.topK"
                  type="number"
                  min="1"
                  max="50"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
            </div>

            <!-- Rerank option -->
            <div class="flex items-center">
              <input
                v-model="searchConfig.rerank"
                type="checkbox"
                id="rerank"
                class="h-4 w-4 text-indigo-600 border-gray-300 rounded"
              />
              <label for="rerank" class="ml-2 text-sm text-gray-700">
                Enable reranking (requires Cohere API key)
              </label>
            </div>

            <!-- Search input -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">Search Query</label>
              <div class="flex gap-2">
                <input
                  v-model="searchQuery"
                  @keyup.enter="performSearch"
                  type="text"
                  placeholder="Enter your search query..."
                  class="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
                <button
                  @click="performSearch"
                  :disabled="isSearching || !searchQuery.trim()"
                  class="px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {{ isSearching ? 'Searching...' : 'Search' }}
                </button>
              </div>
            </div>

            <!-- Search results -->
            <div v-if="searchResults.length > 0" class="mt-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">
                Results ({{ searchResults.length }})
              </h3>
              <div class="space-y-4">
                <div
                  v-for="(result, idx) in searchResults"
                  :key="idx"
                  class="p-4 border border-gray-200 rounded-lg hover:border-indigo-300 transition-colors"
                >
                  <div class="flex justify-between items-start mb-2">
                    <span class="text-xs font-semibold text-indigo-600">
                      Rank #{{ result.rank }}
                    </span>
                    <div class="flex gap-4 text-xs text-gray-500">
                      <span>Score: {{ result.score.toFixed(4) }}</span>
                      <span v-if="result.search_method">Method: {{ result.search_method }}</span>
                    </div>
                  </div>
                  <p class="text-gray-800 mb-2">{{ result.text }}</p>
                  <div class="text-xs text-gray-500">
                    <span v-if="result.metadata.filename">File: {{ result.metadata.filename }}</span>
                    <span v-if="result.metadata.document_id" class="ml-4">
                      Doc ID: {{ result.metadata.document_id }}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <!-- No results -->
            <div v-else-if="hasSearched" class="text-center text-gray-500 py-8">
              No results found
            </div>
          </div>

          <!-- Upload Tab -->
          <div v-if="activeTab === 'upload'" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <!-- Collection -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Collection</label>
                <select v-model="uploadConfig.collectionId" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option v-for="col in collections" :key="col.id" :value="col.id">
                    {{ col.name }}
                  </option>
                </select>
              </div>

              <!-- Embedding Provider -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Embedding Provider</label>
                <select v-model="uploadConfig.embeddingProviderId" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option v-for="provider in embeddingProviders" :key="provider.id" :value="provider.id">
                    {{ provider.name }}
                  </option>
                </select>
              </div>

              <!-- Chunking Strategy -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Chunking Strategy</label>
                <select v-model="uploadConfig.chunkingStrategyId" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option v-for="strategy in chunkingStrategies" :key="strategy.id" :value="strategy.id">
                    {{ strategy.name }}
                  </option>
                </select>
              </div>
            </div>

            <!-- File upload -->
            <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-400 transition-colors">
              <input
                ref="fileInput"
                type="file"
                @change="handleFileSelect"
                accept=".pdf,.docx,.txt,.html,.md"
                class="hidden"
              />
              <button
                @click="$refs.fileInput.click()"
                class="px-6 py-3 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
              >
                Select File
              </button>
              <p class="mt-2 text-sm text-gray-500">
                Supported: PDF, DOCX, TXT, HTML, MD
              </p>
              <p v-if="selectedFile" class="mt-2 text-sm font-medium text-gray-700">
                Selected: {{ selectedFile.name }}
              </p>
            </div>

            <button
              @click="uploadDocument"
              :disabled="!selectedFile || isUploading"
              class="w-full px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {{ isUploading ? 'Uploading...' : 'Upload and Process' }}
            </button>

            <!-- Uploaded documents -->
            <div v-if="documents.length > 0" class="mt-8">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Uploaded Documents</h3>
              <div class="space-y-2">
                <div
                  v-for="doc in documents"
                  :key="doc.id"
                  class="flex justify-between items-center p-3 bg-gray-50 rounded-lg"
                >
                  <div>
                    <p class="font-medium text-gray-900">{{ doc.filename }}</p>
                    <p class="text-sm text-gray-500">
                      {{ doc.chunk_count }} chunks • Status: {{ doc.status }}
                    </p>
                  </div>
                  <button
                    @click="deleteDocument(doc.id)"
                    class="px-3 py-1 text-sm text-red-600 hover:bg-red-50 rounded"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Crawl Tab -->
          <div v-if="activeTab === 'crawl'" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <!-- URL input -->
              <div class="md:col-span-2">
                <label class="block text-sm font-medium text-gray-700 mb-2">Website URL</label>
                <input
                  v-model="crawlConfig.url"
                  type="url"
                  placeholder="https://example.com"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>

              <!-- Crawl type -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Crawl Type</label>
                <select v-model="crawlConfig.crawlType" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option value="single">Single Page</option>
                  <option value="sitemap">Sitemap</option>
                  <option value="recursive">Recursive (Follow Links)</option>
                </select>
              </div>

              <!-- Max pages -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Max Pages</label>
                <input
                  v-model.number="crawlConfig.maxPages"
                  type="number"
                  min="1"
                  max="1000"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>

              <!-- Collection -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Collection</label>
                <select v-model="crawlConfig.collectionId" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option v-for="col in collections" :key="col.id" :value="col.id">
                    {{ col.name }}
                  </option>
                </select>
              </div>

              <!-- Embedding Provider -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Embedding Provider</label>
                <select v-model="crawlConfig.embeddingProviderId" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                  <option v-for="provider in embeddingProviders" :key="provider.id" :value="provider.id">
                    {{ provider.name }}
                  </option>
                </select>
              </div>
            </div>

            <button
              @click="startCrawl"
              :disabled="!crawlConfig.url || isCrawling"
              class="w-full px-6 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {{ isCrawling ? 'Crawling...' : 'Start Crawl' }}
            </button>

            <!-- Crawl jobs -->
            <div v-if="crawlJobs.length > 0" class="mt-8">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Crawl Jobs</h3>
              <div class="space-y-2">
                <div
                  v-for="job in crawlJobs"
                  :key="job.id"
                  class="p-3 bg-gray-50 rounded-lg"
                >
                  <div class="flex justify-between items-start">
                    <div class="flex-1">
                      <p class="font-medium text-gray-900">{{ job.url }}</p>
                      <p class="text-sm text-gray-500">
                        Status: {{ job.status }} • Type: {{ job.crawl_type }}
                      </p>
                      <p class="text-sm text-gray-500">
                        Pages: {{ job.pages_crawled }} crawled, {{ job.pages_failed }} failed
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Config Tab -->
          <div v-if="activeTab === 'config'" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <!-- Collections -->
              <div>
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Collections</h3>
                <div class="space-y-2">
                  <div
                    v-for="col in collections"
                    :key="col.id"
                    class="p-3 bg-gray-50 rounded-lg"
                  >
                    <p class="font-medium">{{ col.name }}</p>
                    <p class="text-sm text-gray-500">{{ col.description || 'No description' }}</p>
                  </div>
                </div>
                
                <!-- Create collection -->
                <div class="mt-4 p-4 border border-gray-200 rounded-lg">
                  <h4 class="font-medium mb-2">Create Collection</h4>
                  <input
                    v-model="newCollection.name"
                    type="text"
                    placeholder="Collection name"
                    class="w-full px-3 py-2 border border-gray-300 rounded-md mb-2"
                  />
                  <input
                    v-model="newCollection.description"
                    type="text"
                    placeholder="Description (optional)"
                    class="w-full px-3 py-2 border border-gray-300 rounded-md mb-2"
                  />
                  <button
                    @click="createCollection"
                    :disabled="!newCollection.name"
                    class="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400"
                  >
                    Create
                  </button>
                </div>
              </div>

              <!-- Providers & Strategies -->
              <div>
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Embedding Providers</h3>
                <div class="space-y-2 mb-6">
                  <div
                    v-for="provider in embeddingProviders"
                    :key="provider.id"
                    class="p-3 bg-gray-50 rounded-lg"
                  >
                    <p class="font-medium">{{ provider.name }}</p>
                    <p class="text-xs text-gray-500">
                      {{ provider.provider_type }} • Dimension: {{ provider.dimension }}
                    </p>
                  </div>
                </div>

                <h3 class="text-lg font-semibold text-gray-900 mb-4">Chunking Strategies</h3>
                <div class="space-y-2">
                  <div
                    v-for="strategy in chunkingStrategies"
                    :key="strategy.id"
                    class="p-3 bg-gray-50 rounded-lg"
                  >
                    <p class="font-medium">{{ strategy.name }}</p>
                    <p class="text-xs text-gray-500">
                      Size: {{ strategy.chunk_size }} • Overlap: {{ strategy.overlap }}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

const API_BASE = '/api/rag'

// Tabs
const tabs = [
  { id: 'search', name: 'Search' },
  { id: 'upload', name: 'Upload Documents' },
  { id: 'crawl', name: 'Web Crawl' },
  { id: 'config', name: 'Configuration' }
]
const activeTab = ref('search')

// Data
const collections = ref<any[]>([])
const embeddingProviders = ref<any[]>([])
const chunkingStrategies = ref<any[]>([])
const documents = ref<any[]>([])
const crawlJobs = ref<any[]>([])

// Search
const searchQuery = ref('')
const searchResults = ref<any[]>([])
const hasSearched = ref(false)
const isSearching = ref(false)
const searchConfig = ref({
  collectionId: 1,
  embeddingProviderId: 1,
  topK: 10,
  searchType: 'semantic',
  rerank: false
})

// Upload
const selectedFile = ref<File | null>(null)
const isUploading = ref(false)
const uploadConfig = ref({
  collectionId: 1,
  embeddingProviderId: 1,
  chunkingStrategyId: 1
})
const fileInput = ref<HTMLInputElement>()

// Crawl
const isCrawling = ref(false)
const crawlConfig = ref({
  url: '',
  crawlType: 'single',
  maxPages: 100,
  maxDepth: 2,
  collectionId: 1,
  embeddingProviderId: 1,
  chunkingStrategyId: 1
})

// New collection
const newCollection = ref({
  name: '',
  description: ''
})

// Methods
const loadData = async () => {
  try {
    const [colsRes, providersRes, strategiesRes, docsRes, jobsRes] = await Promise.all([
      axios.get(`${API_BASE}/collections`),
      axios.get(`${API_BASE}/embedding-providers`),
      axios.get(`${API_BASE}/chunking-strategies`),
      axios.get(`${API_BASE}/documents`),
      axios.get(`${API_BASE}/crawl-jobs`)
    ])
    
    collections.value = colsRes.data
    embeddingProviders.value = providersRes.data
    chunkingStrategies.value = strategiesRes.data
    documents.value = docsRes.data
    crawlJobs.value = jobsRes.data
  } catch (error) {
    console.error('Error loading data:', error)
  }
}

const performSearch = async () => {
  if (!searchQuery.value.trim()) return
  
  isSearching.value = true
  hasSearched.value = false
  
  try {
    const response = await axios.post(`${API_BASE}/search`, {
      query: searchQuery.value,
      collection_id: searchConfig.value.collectionId,
      embedding_provider_id: searchConfig.value.embeddingProviderId,
      top_k: searchConfig.value.topK,
      search_type: searchConfig.value.searchType,
      rerank: searchConfig.value.rerank
    })
    
    searchResults.value = response.data.results
    hasSearched.value = true
  } catch (error) {
    console.error('Search error:', error)
    alert('Search failed. See console for details.')
  } finally {
    isSearching.value = false
  }
}

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    selectedFile.value = target.files[0]
  }
}

const uploadDocument = async () => {
  if (!selectedFile.value) return
  
  isUploading.value = true
  
  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    
    await axios.post(
      `${API_BASE}/documents/upload?collection_id=${uploadConfig.value.collectionId}&embedding_provider_id=${uploadConfig.value.embeddingProviderId}&chunking_strategy_id=${uploadConfig.value.chunkingStrategyId}`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' }
      }
    )
    
    alert('Document uploaded successfully!')
    selectedFile.value = null
    await loadData()
  } catch (error) {
    console.error('Upload error:', error)
    alert('Upload failed. See console for details.')
  } finally {
    isUploading.value = false
  }
}

const deleteDocument = async (docId: number) => {
  if (!confirm('Delete this document?')) return
  
  try {
    await axios.delete(`${API_BASE}/documents/${docId}`)
    await loadData()
  } catch (error) {
    console.error('Delete error:', error)
    alert('Delete failed.')
  }
}

const startCrawl = async () => {
  if (!crawlConfig.value.url) return
  
  isCrawling.value = true
  
  try {
    await axios.post(`${API_BASE}/crawl`, {
      url: crawlConfig.value.url,
      crawl_type: crawlConfig.value.crawlType,
      max_pages: crawlConfig.value.maxPages,
      max_depth: crawlConfig.value.maxDepth,
      collection_id: crawlConfig.value.collectionId,
      embedding_provider_id: crawlConfig.value.embeddingProviderId,
      chunking_strategy_id: crawlConfig.value.chunkingStrategyId
    })
    
    alert('Crawl started! Check the jobs list for progress.')
    await loadData()
  } catch (error) {
    console.error('Crawl error:', error)
    alert('Crawl failed. See console for details.')
  } finally {
    isCrawling.value = false
  }
}

const createCollection = async () => {
  if (!newCollection.value.name) return
  
  try {
    await axios.post(`${API_BASE}/collections`, newCollection.value)
    newCollection.value = { name: '', description: '' }
    await loadData()
  } catch (error) {
    console.error('Create collection error:', error)
    alert('Failed to create collection.')
  }
}

onMounted(() => {
  loadData()
})
</script>
