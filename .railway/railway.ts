import { defineRailway, project, service } from "railway/iac";

export default defineRailway(() => {
  const dashboard = service("aml-defense-dashboard", {
    dockerfile: "deployment/Dockerfile",
    start: "streamlit run src/dashboard/streamlit_app.py --server.port $PORT --server.address 0.0.0.0",
    env: {
      AUTH_USERNAME: "admin",
      AUTH_PASSWORD: "ciciomt2024",
    },
  });

  return project("aml-defense-dashboard", {
    resources: [dashboard],
  });
});
