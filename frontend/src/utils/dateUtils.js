/**
 * Safely parses any date/time string, timestamp, or Date object into a valid JavaScript Date.
 * Handles ISO strings with or without 'Z', UTC timestamps, and fallback handling.
 */
export const parseUtcDate = (val) => {
  if (!val) return null;
  if (val instanceof Date) return isNaN(val.getTime()) ? null : val;
  if (typeof val === 'number') {
    const d = new Date(val);
    return isNaN(d.getTime()) ? null : d;
  }
  if (typeof val === 'string') {
    let str = val.trim();
    if (!str) return null;
    // If ISO date string without timezone indicator (e.g. "2026-10-12T12:27:30.412000"),
    // append 'Z' so it is parsed as UTC rather than local time.
    if (str.includes('T') && !str.endsWith('Z') && !/[+-]\d{2}:?\d{2}$/.test(str)) {
      str = str + 'Z';
    }
    const d = new Date(str);
    return isNaN(d.getTime()) ? null : d;
  }
  return null;
};

/**
 * Returns human-friendly relative time (e.g. "Just now", "5 mins ago", "2 hours ago", "Yesterday", "3 days ago").
 */
export const formatTimeAgo = (rawDate) => {
  const date = parseUtcDate(rawDate);
  if (!date) return 'Recently';

  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.round(diffMs / 1000);
  const diffMins = Math.round(diffMs / 60000);
  const diffHours = Math.round(diffMs / 3600000);
  const diffDays = Math.round(diffMs / 86400000);

  if (diffSec < 60 && diffSec >= -30) {
    return 'Just now';
  }
  if (diffMins < 60 && diffMins > 0) {
    return `${diffMins} min${diffMins === 1 ? '' : 's'} ago`;
  }
  if (diffHours < 24 && diffHours > 0) {
    return `${diffHours} hr${diffHours === 1 ? '' : 's'} ago`;
  }
  if (diffDays === 1) {
    return 'Yesterday';
  }
  if (diffDays > 1 && diffDays < 30) {
    return `${diffDays} days ago`;
  }
  if (diffDays >= 30) {
    const diffMonths = Math.round(diffDays / 30);
    return `${diffMonths} mo${diffMonths === 1 ? '' : 's'} ago`;
  }
  return 'Just now';
};

/**
 * Formats a date for document cards:
 * returns { formattedDate, formattedTime, timeAgo, fullDateTime }
 */
export const formatAuditCardDate = (rawDate) => {
  const date = parseUtcDate(rawDate);
  if (!date) {
    return {
      formattedDate: 'N/A',
      formattedTime: '',
      timeAgo: 'Just now',
      fullDateTime: 'Unknown date',
    };
  }

  const formattedDate = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  const formattedTime = date.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });

  const timeAgo = formatTimeAgo(date);

  return {
    formattedDate,
    formattedTime,
    timeAgo,
    fullDateTime: `${formattedDate} at ${formattedTime}`,
  };
};
