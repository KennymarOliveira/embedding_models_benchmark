from app.core.config import (
    AVAILABLE_MODELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
    GPU_DEVICE_NAME,
    GPU_VRAM_TOTAL_MB,
    HAS_CUDA,
)


def get_dashboard_html() -> str:
    """Gera o HTML do painel interativo com caixas de seleção dinâmicas para os modelos."""

    model_checkboxes = ""
    for key, cfg in AVAILABLE_MODELS.items():
        is_checked = "checked" if cfg.get("enabled", True) else ""
        badge_color = "#3b82f6" if "8b" in key else "#10b981" if "bge" in key else "#8b5cf6" if "e5" in key else "#f59e0b"
        model_checkboxes += f"""
        <label class="model-card" for="model_{key}">
            <div class="checkbox-wrapper">
                <input type="checkbox" id="model_{key}" name="models" value="{key}" {is_checked}>
                <span class="custom-checkbox"></span>
            </div>
            <div class="card-details">
                <div class="card-header">
                    <span class="card-title">{cfg['display_name']}</span>
                    <span class="card-badge" style="background-color: {badge_color}20; color: {badge_color}; border-color: {badge_color}40;">
                        {cfg.get('class_key', 'embedding')}
                    </span>
                </div>
                <div class="card-id">{cfg['model_id']}</div>
                <p class="card-desc">{cfg.get('description', '')}</p>
                <div class="card-meta">
                    <span>📏 Max Seq: <strong>{cfg.get('max_seq_length', 512)}</strong></span>
                    <span>⚙️ Dtype: <strong>{cfg.get('torch_dtype', 'float32')}</strong></span>
                    <span>📦 Batch: <strong>{cfg.get('default_batch_size', 8)}</strong></span>
                </div>
            </div>
        </label>
        """

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark de Modelos de Embedding Local</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #273549;
            --bg-input: #0f172a;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --border-color: #334155;
            --border-focus: #60a5fa;
            --radius-md: 10px;
            --radius-lg: 14px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 30px 20px;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }}

        .title-group h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .title-group p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 4px;
        }}

        .hw-badge {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 8px 14px;
            border-radius: var(--radius-md);
            font-size: 0.85rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .hw-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: {"#10b981" if HAS_CUDA else "#f59e0b"};
        }}

        .section-box {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 24px;
        }}

        .section-title {{
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        /* Upload Area */
        .drop-zone {{
            border: 2px dashed var(--border-color);
            border-radius: var(--radius-md);
            padding: 32px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background-color: var(--bg-primary);
        }}

        .drop-zone:hover, .drop-zone.dragover {{
            border-color: var(--accent-blue);
            background-color: #1e293b80;
        }}

        .drop-zone svg {{
            width: 44px;
            height: 44px;
            color: var(--accent-blue);
            margin-bottom: 10px;
        }}

        .file-info {{
            margin-top: 10px;
            font-weight: 500;
            color: #60a5fa;
            word-break: break-all;
        }}

        /* Model Cards & Checkboxes */
        .models-toolbar {{
            display: flex;
            gap: 10px;
        }}

        .btn-link {{
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: 0.15s;
        }}

        .btn-link:hover {{
            background: var(--bg-card);
            color: var(--text-primary);
        }}

        .models-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 14px;
            margin-top: 14px;
        }}

        .model-card {{
            background-color: var(--bg-card);
            border: 2px solid transparent;
            border-radius: var(--radius-md);
            padding: 16px;
            display: flex;
            gap: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }}

        .model-card:hover {{
            border-color: #475569;
            transform: translateY(-2px);
        }}

        .model-card:has(input:checked) {{
            border-color: var(--accent-blue);
            background-color: #1e3a5f40;
        }}

        .checkbox-wrapper {{
            display: flex;
            align-items: flex-start;
            padding-top: 2px;
        }}

        .model-card input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            accent-color: var(--accent-blue);
            cursor: pointer;
        }}

        .card-details {{
            flex: 1;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-title {{
            font-weight: 600;
            font-size: 1rem;
        }}

        .card-badge {{
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 9999px;
            border: 1px solid;
            text-transform: uppercase;
        }}

        .card-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin: 3px 0 6px;
        }}

        .card-desc {{
            font-size: 0.82rem;
            color: #cbd5e1;
            margin-bottom: 8px;
        }}

        .card-meta {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}

        /* Parameters Grid */
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }}

        .form-group label {{
            display: block;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}

        .form-group input, .form-group select {{
            width: 100%;
            background-color: var(--bg-input);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
        }}

        .form-group input:focus, .form-group select:focus {{
            border-color: var(--border-focus);
        }}

        /* Submit Button */
        .btn-submit {{
            width: 100%;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            padding: 14px 20px;
            border: none;
            border-radius: var(--radius-md);
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}

        .btn-submit:hover:not(:disabled) {{
            filter: brightness(1.1);
            transform: translateY(-1px);
        }}

        .btn-submit:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
        }}

        /* Spinner */
        .spinner {{
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            display: none;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Results Section */
        #results-container {{
            display: none;
        }}

        .rankings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }}

        .rank-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 16px;
            text-align: center;
        }}

        .rank-icon {{
            font-size: 1.6rem;
            margin-bottom: 4px;
        }}

        .rank-label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-weight: 600;
        }}

        .rank-winner {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #60a5fa;
            margin-top: 4px;
        }}

        /* Results Table */
        .table-wrapper {{
            overflow-x: auto;
            margin-bottom: 24px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }}

        th, td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: var(--bg-card);
            color: var(--text-secondary);
            font-weight: 600;
        }}

        tr:hover td {{
            background-color: #1e293b50;
        }}

        .chart-box {{
            background-color: var(--bg-card);
            border-radius: var(--radius-md);
            padding: 16px;
            margin-top: 20px;
        }}

        .actions-bar {{
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            margin-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-group">
                <h1>⚡ Benchmark de Embeddings Local</h1>
                <p>Medição de latência, vazão e uso de recursos de hardware em modelos de embedding</p>
            </div>
            <div class="hw-badge">
                <span class="hw-dot"></span>
                <span>{GPU_DEVICE_NAME} ({GPU_VRAM_TOTAL_MB} MB)</span>
                <a href="/docs" target="_blank" style="color: var(--accent-blue); text-decoration: none; margin-left: 10px; font-weight: 600;">Swagger Docs ↗</a>
            </div>
        </header>

        <form id="benchmark-form">
            <!-- Passo 1: Upload do Documento -->
            <div class="section-box">
                <div class="section-title">
                    <span>1. Envie o Documento</span>
                    <span style="font-size: 0.8rem; color: var(--text-secondary);">Formatos: PDF, DOCX, DOC, ODT, TXT</span>
                </div>
                <div class="drop-zone" id="drop-zone">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                    </svg>
                    <p style="font-weight: 500;">Arraste e solte o arquivo aqui ou clique para selecionar</p>
                    <input type="file" id="file-input" name="file" accept=".pdf,.docx,.doc,.odt,.txt" style="display: none;" required>
                    <div class="file-info" id="file-info">Nenhum arquivo selecionado</div>
                </div>
            </div>

            <!-- Passo 2: Caixas de Seleção dos Modelos -->
            <div class="section-box">
                <div class="section-title">
                    <span>2. Selecione os Modelos para Avaliar</span>
                    <div class="models-toolbar">
                        <button type="button" class="btn-link" onclick="toggleAllModels(true)">Selecionar Todos</button>
                        <button type="button" class="btn-link" onclick="toggleAllModels(false)">Desmarcar Todos</button>
                    </div>
                </div>
                <div class="models-grid">
                    {model_checkboxes}
                </div>
            </div>

            <!-- Passo 3: Parâmetros Opcionais -->
            <div class="section-box">
                <div class="section-title">
                    <span>3. Parâmetros de Execução</span>
                </div>
                <div class="params-grid">
                    <div class="form-group">
                        <label for="batch_size">Batch Size</label>
                        <input type="number" id="batch_size" name="batch_size" value="{DEFAULT_BATCH_SIZE}" min="1" max="128">
                    </div>
                    <div class="form-group">
                        <label for="chunk_size">Tamanho do Chunk (chars)</label>
                        <input type="number" id="chunk_size" name="chunk_size" value="{DEFAULT_CHUNK_SIZE}" min="50" max="10000">
                    </div>
                    <div class="form-group">
                        <label for="chunk_overlap">Overlap (chars)</label>
                        <input type="number" id="chunk_overlap" name="chunk_overlap" value="{DEFAULT_CHUNK_OVERLAP}" min="0" max="500">
                    </div>
                    <div class="form-group">
                        <label for="chunk_strategy">Estratégia de Chunking</label>
                        <select id="chunk_strategy" name="chunk_strategy">
                            <option value="paragraph" {"selected" if DEFAULT_CHUNK_STRATEGY == "paragraph" else ""}>Parágrafos (Recomendado)</option>
                            <option value="fixed" {"selected" if DEFAULT_CHUNK_STRATEGY == "fixed" else ""}>Janela Fixa (com Overlap)</option>
                            <option value="sentence" {"selected" if DEFAULT_CHUNK_STRATEGY == "sentence" else ""}>Sentenças</option>
                        </select>
                    </div>
                </div>
            </div>

            <button type="submit" class="btn-submit" id="submit-btn">
                <span class="spinner" id="btn-spinner"></span>
                <span id="btn-text">🚀 Executar Benchmark</span>
            </button>
        </form>

        <!-- Resultados -->
        <div class="section-box" id="results-container" style="margin-top: 30px;">
            <div class="section-title">
                <span>Resultados do Benchmark</span>
                <span id="doc-meta-badge" style="font-size: 0.85rem; color: #60a5fa; font-weight: normal;"></span>
            </div>

            <div class="rankings-grid" id="rankings-grid"></div>

            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Modelo</th>
                            <th>Tempo Total</th>
                            <th>Latência / Chunk</th>
                            <th>Throughput</th>
                            <th>Dimensão</th>
                            <th>Pico VRAM</th>
                            <th>Delta RAM</th>
                            <th>Cold Start</th>
                        </tr>
                    </thead>
                    <tbody id="results-table-body"></tbody>
                </table>
            </div>

            <div class="chart-box">
                <canvas id="chart-comparison" style="max-height: 380px;"></canvas>
            </div>

            <div class="actions-bar">
                <button type="button" class="btn-link" onclick="downloadJSON()">📥 Baixar Relatório JSON</button>
            </div>
        </div>
    </div>

    <script>
        let lastResultData = null;
        let chartInstance = null;

        // Upload Drag & Drop
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const fileInfo = document.getElementById('file-info');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {{ e.preventDefault(); dropZone.classList.add('dragover'); }});
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {{
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {{
                fileInput.files = e.dataTransfer.files;
                updateFileInfo();
            }}
        }});
        fileInput.addEventListener('change', updateFileInfo);

        function updateFileInfo() {{
            if (fileInput.files.length > 0) {{
                const file = fileInput.files[0];
                const sizeKb = (file.size / 1024).toFixed(1);
                fileInfo.innerHTML = `📄 <strong>${{file.name}}</strong> (${{sizeKb}} KB)`;
            }} else {{
                fileInfo.textContent = "Nenhum arquivo selecionado";
            }}
        }}

        function toggleAllModels(checked) {{
            document.querySelectorAll('input[name="models"]').forEach(cb => cb.checked = checked);
        }}

        // Submissão do Formulário
        document.getElementById('benchmark-form').addEventListener('submit', async (e) => {{
            e.preventDefault();
            const checkedBoxes = document.querySelectorAll('input[name="models"]:checked');
            if (checkedBoxes.length === 0) {{
                alert("Por favor, selecione ao menos um modelo nas caixas de seleção!");
                return;
            }}

            const submitBtn = document.getElementById('submit-btn');
            const btnSpinner = document.getElementById('btn-spinner');
            const btnText = document.getElementById('btn-text');

            submitBtn.disabled = true;
            btnSpinner.style.display = "inline-block";
            btnText.textContent = "Processando e vetorizando chunks...";

            const formData = new FormData(e.target);

            try {{
                const response = await fetch('/api/v1/benchmark/run', {{
                    method: 'POST',
                    body: formData
                }});

                if (!response.ok) {{
                    const err = await response.json();
                    throw new Error(err.detail || "Erro durante o benchmark.");
                }}

                const data = await response.json();
                lastResultData = data;
                renderResults(data);
            }} catch (error) {{
                alert("Falha: " + error.message);
            }} finally {{
                submitBtn.disabled = false;
                btnSpinner.style.display = "none";
                btnText.textContent = "🚀 Executar Benchmark";
            }}
        }});

        function renderResults(data) {{
            const container = document.getElementById('results-container');
            container.style.display = "block";
            container.scrollIntoView({{ behavior: 'smooth' }});

            // Metadata
            document.getElementById('doc-meta-badge').textContent = 
                `${{data.document_info.filename}} | ${{data.document_info.total_characters}} caracteres | ~${{data.document_info.total_estimated_tokens}} tokens | ${{data.document_info.total_chunks}} chunks`;

            // Rankings
            const rankGrid = document.getElementById('rankings-grid');
            rankGrid.innerHTML = `
                <div class="rank-card">
                    <div class="rank-icon">⚡</div>
                    <div class="rank-label">Mais Rápido (Tempo)</div>
                    <div class="rank-winner">${{data.rankings.fastest_total_time || "N/A"}}</div>
                </div>
                <div class="rank-card">
                    <div class="rank-icon">🚀</div>
                    <div class="rank-label">Maior Throughput</div>
                    <div class="rank-winner">${{data.rankings.highest_token_throughput || "N/A"}}</div>
                </div>
                <div class="rank-card">
                    <div class="rank-icon">⏱️</div>
                    <div class="rank-label">Menor Latência / Chunk</div>
                    <div class="rank-winner">${{data.rankings.lowest_latency_per_chunk || "N/A"}}</div>
                </div>
                <div class="rank-card">
                    <div class="rank-icon">💾</div>
                    <div class="rank-label">Mais Econômico em VRAM</div>
                    <div class="rank-winner">${{data.rankings.lowest_vram_usage || "N/A"}}</div>
                </div>
            `;

            // Table
            const tbody = document.getElementById('results-table-body');
            tbody.innerHTML = "";
            const labels = [];
            const times = [];
            const tokens = [];

            for (const [key, res] of Object.entries(data.results)) {{
                if (res.error) {{
                    tbody.innerHTML += `<tr>
                        <td><strong>${{res.display_name}}</strong></td>
                        <td colspan="7" style="color: #ef4444;">Erro: ${{res.error}}</td>
                    </tr>`;
                    continue;
                }}

                labels.push(res.display_name);
                times.push(res.total_inference_time_seconds);
                tokens.push(res.throughput_tokens_per_sec);

                tbody.innerHTML += `<tr>
                    <td><strong>${{res.display_name}}</strong><br><span style="font-size:0.75rem; color:#94a3b8;">${{res.model_id}}</span></td>
                    <td><strong>${{res.total_inference_time_seconds.toFixed(3)}} s</strong> (${{res.total_inference_time_ms.toFixed(1)}} ms)</td>
                    <td>${{res.avg_chunk_latency_ms.toFixed(1)}} ms</td>
                    <td><strong style="color:#10b981;">${{res.throughput_tokens_per_sec.toFixed(0)}}</strong> tok/s</td>
                    <td>${{res.embedding_dimension}}</td>
                    <td>${{res.gpu_peak_mb.toFixed(1)}} MB</td>
                    <td>+${{res.ram_delta_mb.toFixed(1)}} MB <span style="font-size:0.75rem; color:#94a3b8;">(${{res.ram_after_mb.toFixed(0)}} MB total)</span></td>
                    <td>${{res.load_time_seconds.toFixed(2)}} s</td>
                </tr>`;
            }}

            // Chart
            renderChart(labels, times, tokens);
        }}

        function renderChart(labels, times, tokens) {{
            const ctx = document.getElementById('chart-comparison').getContext('2d');
            if (chartInstance) chartInstance.destroy();

            chartInstance = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Tempo Total de Inferência (s)',
                            data: times,
                            backgroundColor: '#3b82f6',
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Throughput (Tokens / s)',
                            data: tokens,
                            backgroundColor: '#10b981',
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    interaction: {{ mode: 'index', intersect: false }},
                    scales: {{
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{ display: true, text: 'Segundos (s)', color: '#94a3b8' }},
                            grid: {{ color: '#334155' }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{ display: true, text: 'Tokens / s', color: '#94a3b8' }},
                            grid: {{ drawOnChartArea: false }}
                        }},
                        x: {{
                            grid: {{ color: '#334155' }}
                        }}
                    }}
                }}
            }});
        }}

        function downloadJSON() {{
            if (!lastResultData) return;
            const blob = new Blob([JSON.stringify(lastResultData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `benchmark_${{lastResultData.benchmark_id}}.json`;
            a.click();
        }}
    </script>
</body>
</html>
    """

