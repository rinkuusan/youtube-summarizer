package jp.tanakabutton.videonotes;
public final class YouTubeUrlsTest {
 public static void main(String[] args) {
  require(YouTubeUrls.extract("タイトル https://youtu.be/jNQXAC9IVRw?si=abc").size()==1,"shared title");
  require(YouTubeUrls.extract("https://youtu.be/jNQXAC9IVRw\nhttps://www.youtube.com/watch?v=jNQXAC9IVRw&t=5").size()==1,"dedup");
  require(YouTubeUrls.extract("https://youtube.com.evil.test/watch?v=jNQXAC9IVRw").isEmpty(),"host spoof");
  require(YouTubeUrls.extract("https://evil@youtube.com/watch?v=jNQXAC9IVRw").isEmpty(),"userinfo");
  require(YouTubeUrls.extract("https://youtube.com/shorts/jNQXAC9IVRw https://youtube.com/live/aaaaaaaaaaa").size()==2,"shorts/live");
  require(YouTubeUrls.extract("これは秘密の文章です").isEmpty(),"non URL clipboard");
  require(YouTubeUrls.extract("（https://youtu.be/jNQXAC9IVRw)").size()==1,"closing bracket");
  require(YouTubeUrls.extract("https://youtube.com/watch?feature=share&v=jNQXAC9IVRw").size()==1,"query order");
  StringBuilder many=new StringBuilder();for(int i=0;i<25;i++)many.append("https://youtu.be/"+String.format("%011d",i)+" ");
  require(YouTubeUrls.extract(many.toString()).size()==20,"input bound");
  System.out.println("PASS: shared-text extraction, normalized duplicates, deceptive hosts, Shorts/live, non-URL clipboard, punctuation, bounds");
 }
 static void require(boolean value,String test){if(!value)throw new AssertionError(test);}
}
