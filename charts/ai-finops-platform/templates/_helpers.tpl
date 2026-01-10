{{/*
Expand the name of the chart.
*/}}
{{- define "ai-finops.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ai-finops.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ai-finops.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ai-finops.labels" -}}
helm.sh/chart: {{ include "ai-finops.chart" . }}
{{ include "ai-finops.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ai-finops.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ai-finops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
GPU Enricher labels
*/}}
{{- define "ai-finops.gpuEnricher.labels" -}}
{{ include "ai-finops.labels" . }}
app.kubernetes.io/component: gpu-enricher
{{- end }}

{{- define "ai-finops.gpuEnricher.selectorLabels" -}}
app.kubernetes.io/name: gpu-enricher
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Prometheus labels
*/}}
{{- define "ai-finops.prometheus.labels" -}}
{{ include "ai-finops.labels" . }}
app.kubernetes.io/component: prometheus
{{- end }}

{{- define "ai-finops.prometheus.selectorLabels" -}}
app.kubernetes.io/name: prometheus
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Grafana labels
*/}}
{{- define "ai-finops.grafana.labels" -}}
{{ include "ai-finops.labels" . }}
app.kubernetes.io/component: grafana
{{- end }}

{{- define "ai-finops.grafana.selectorLabels" -}}
app.kubernetes.io/name: grafana
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
OpenCost labels
*/}}
{{- define "ai-finops.opencost.labels" -}}
{{ include "ai-finops.labels" . }}
app.kubernetes.io/component: opencost
{{- end }}

{{- define "ai-finops.opencost.selectorLabels" -}}
app.kubernetes.io/name: opencost
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Alertmanager labels
*/}}
{{- define "ai-finops.alertmanager.labels" -}}
{{ include "ai-finops.labels" . }}
app.kubernetes.io/component: alertmanager
{{- end }}

{{- define "ai-finops.alertmanager.selectorLabels" -}}
app.kubernetes.io/name: alertmanager
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "ai-finops.namespace" -}}
{{- .Values.global.namespace | default "ai-finops" }}
{{- end }}
