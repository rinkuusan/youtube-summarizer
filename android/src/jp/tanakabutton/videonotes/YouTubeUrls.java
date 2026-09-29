package jp.tanakabutton.videonotes;
import java.net.URI;
import java.net.URLDecoder;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
public final class YouTubeUrls {
    private static final Pattern URL = Pattern.compile("https?://[^\\s<>\\\"\\u3000]+", Pattern.CASE_INSENSITIVE);
    public static LinkedHashSet<String> extract(String text) {
        LinkedHashSet<String> result = new LinkedHashSet<>();
        if (text == null) return result;
        Matcher m = URL.matcher(text.substring(0, Math.min(text.length(), 100000)));
        while (m.find() && result.size() < 20) {
            String raw = m.group().replaceAll("[)\\]}>、。！!？]+$", "");
            String id = id(raw);
            if (id != null) result.add("https://www.youtube.com/watch?v=" + id);
        }
        return result;
    }
    static String id(String raw) {
        try {
            URI uri = new URI(raw); String host = uri.getHost();
            if (host == null || uri.getUserInfo() != null || !("https".equalsIgnoreCase(uri.getScheme()) || "http".equalsIgnoreCase(uri.getScheme()))) return null;
            host = host.toLowerCase(Locale.ROOT); String id = null; String path = uri.getPath();
            if (host.equals("youtu.be")) { String[] parts = path.split("/"); if (parts.length > 1) id = parts[1]; }
            else if (host.equals("youtube.com") || host.equals("www.youtube.com") || host.equals("m.youtube.com") || host.equals("music.youtube.com")) {
                if (path.equals("/watch") && uri.getRawQuery() != null) {
                    for (String pair : uri.getRawQuery().split("&")) { String[] kv = pair.split("=",2); if (kv.length == 2 && kv[0].equals("v")) { id = URLDecoder.decode(kv[1], "UTF-8"); break; } }
                } else if (path.matches("^/(shorts|live|embed)/.*")) { String[] parts = path.split("/"); if (parts.length > 2) id = parts[2]; }
            }
            return id != null && id.matches("[A-Za-z0-9_-]{11}") ? id : null;
        } catch (Exception e) { return null; }
    }
}
