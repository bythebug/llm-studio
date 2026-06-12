export function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    queued:    "bg-gray-100 text-gray-600",
    training:  "bg-blue-50 text-blue-700 animate-pulse",
    completed: "bg-green-50 text-green-700",
    failed:    "bg-red-50 text-red-700",
  };
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-medium ${styles[status] ?? "bg-gray-100 text-gray-600"}`}>
      {status}
    </span>
  );
}
