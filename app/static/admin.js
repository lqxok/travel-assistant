/* 运营数据看板：Vue 3 单文件逻辑（挂在 /admin 页面）。 */
const { createApp } = Vue;

createApp({
  data() {
    return {
      token: localStorage.getItem("travel_token") || "",
      stats: null,
      logs: [],
      error: "",
      loading: false,
    };
  },
  mounted() {
    // 支持 /admin?token=xxx 打开一次并记住
    const urlToken = new URLSearchParams(location.search).get("token");
    if (urlToken) {
      this.token = urlToken.trim();
      this.saveToken();
      history.replaceState(null, "", location.pathname);
    }
    this.load();
  },
  methods: {
    saveToken() {
      localStorage.setItem("travel_token", (this.token || "").trim());
    },
    authHeaders() {
      const token = (this.token || "").trim();
      return token ? { "X-Access-Token": token } : {};
    },
    async load() {
      this.loading = true;
      this.error = "";
      try {
        const [statsResp, logsResp] = await Promise.all([
          fetch("/api/v1/admin/stats?days=7", { headers: this.authHeaders() }),
          fetch("/api/v1/admin/logs?limit=15", { headers: this.authHeaders() }),
        ]);
        if (statsResp.status === 401) {
          this.error = "访问口令错误或未填写";
          this.stats = null;
          return;
        }
        if (!statsResp.ok) {
          this.error = `加载失败（HTTP ${statsResp.status}）`;
          return;
        }
        this.stats = await statsResp.json();
        const logsData = await logsResp.json();
        this.logs = logsData.items || [];
      } catch (err) {
        this.error = `网络错误：${err}`;
      } finally {
        this.loading = false;
      }
    },
    maxOf(dist) {
      const values = Object.values(dist || {});
      return values.length ? Math.max(...values) : 1;
    },
    pct(count, dist) {
      return `${Math.round((count / this.maxOf(dist)) * 100)}%`;
    },
    hasData(dist) {
      return Boolean(dist && Object.keys(dist).length);
    },
    barHeight(count) {
      const counts = (this.stats?.trend || []).map((item) => item.count);
      const max = counts.length ? Math.max(...counts, 1) : 1;
      return `${Math.max(3, Math.round((count / max) * 120))}px`;
    },
  },
}).mount("#app");
